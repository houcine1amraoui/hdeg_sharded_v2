#!/usr/bin/env python3
"""Level 1 qualitative views on a fixed subset of frozen primary R3 blocks."""
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
import uuid
import numpy as np
import yaml
from src.utils.get_folders_utils import get_processed_folder
from src.scientific_validation.v1.v1b_io import discover_shards
from src.scientific_validation.v1.v1b_provenance import sha256_file
from src.scientific_validation.v1.r3_primary import frozen_primary_indices
from src.scientific_validation.v1.r3_pilot import CONDITIONS, LEVELS, positive_int, index_hash
from src.scientific_validation.v1.r3_sensitivity import validate_primary
from src.scientific_validation.v1 import v1_qualitative as helper
from src.scientific_validation.v1.v1_qualitative import select_windows, pooled_pca, OFFSETS
from run_r3_pilot import write_report, peak_rss_mib, sampling_step_ns, read_timestamps, collect_level


def main():
    with open('configs/config.yaml',encoding='utf-8') as f:
        config=yaml.safe_load(f)
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference_dir',required=True,type=Path,help='Completed primary R3 directory')
    parser.add_argument('--project_root_dir')
    args=parser.parse_args()
    if args.project_root_dir is not None: config['project_root_dir']=args.project_root_dir
    base=Path(get_processed_folder(config))
    output=base/'v1_validation'/'qualitative'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8])
    output.mkdir(parents=True)
    started=time.perf_counter()
    report=dict(status='RUNNING',dataset=base.name,configuration=config,levels={},
        measurement_family='descriptive_representation_visualization',
        subset_rule='Offsets 0,42,85,127 from every frozen primary block; 128 windows/condition; 384 total',
        design_status='Exploratory visualization rule fixed after quantitative results; selection uses only indices, not embedding values',
        limitation='Pooled centered PCA on selected evaluation rows is descriptive, not a trained detector. No cluster, actor-discrimination, semantic-coherence, or collapse verdict. Two components may discard substantial variation.',
        memory_measure='Process lifetime peak RSS, including imports and loaded shards')
    path=output/'qualitative_report.json'
    write_report(path,report)
    print(f'Level 1 qualitative output: {output}',flush=True)
    try:
        settings=config['v1_validation']
        seed=config['seed']
        if isinstance(seed,bool) or not isinstance(seed,int) or seed<0:raise ValueError('Invalid seed')
        reference_path=args.reference_dir/'primary_report.json'
        reference=json.loads(reference_path.read_text())
        plans=json.loads((args.reference_dir/'block_plan.json').read_text())
        validate_primary(reference,plans,config,base.name)
        checkpoint=Path(config['hdeg']['hbf']['checkpoint'])
        if not checkpoint.is_absolute():checkpoint=base/'hdeg_checkpoints'/checkpoint
        checkpoint_hash=sha256_file(checkpoint)
        preflight_path=base/'v1_validation/preflight.json'
        preflight=json.loads(preflight_path.read_text())
        if checkpoint_hash!=reference['checkpoint_sha256'] or preflight.get('checkpoint_sha256')!=checkpoint_hash:
            raise ValueError('Checkpoint differs from reference/preflight')
        if preflight.get('status')!='PASS' or sha256_file(preflight_path)!=reference['preflight_sha256']:
            raise ValueError('Preflight differs from completed primary')
        report.update(checkpoint_sha256=checkpoint_hash,preflight_sha256=sha256_file(preflight_path),
            reference_run_id=reference['run_id'],reference_report_sha256=sha256_file(reference_path),
            configuration_sha256=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest(),
            settings=dict(offsets_per_block=OFFSETS.tolist(),windows_per_condition=128,seed=seed,pca_scaling="center_only"))
        # Record every directly used project module; preserve the inherited I/O helper.
        import run_r3_pilot as pilot
        if sha256_file(Path(pilot.__file__))!=reference['source_code_sha256']['run_r3_pilot.py']:
            raise ValueError('Pilot I/O helper changed since primary')
        report['source_code_sha256']={str(Path(__file__).name):sha256_file(Path(__file__)),
                                     'v1_qualitative.py':sha256_file(Path(helper.__file__))}
        import inspect
        for fn in (collect_level,discover_shards,sha256_file,frozen_primary_indices,get_processed_folder,validate_primary):
            source=Path(inspect.getsourcefile(fn))
            report['source_code_sha256'][str(source)]=sha256_file(source)
        selected={}; selected_times={}; manifests=[]; step=sampling_step_ns(config['preprocessing']['downsample_freq'])
        dims=reference['dimensions'];dimensions=tuple(dims[k] for k in ('N','K','D','W'))
        for condition,split in CONDITIONS.items():
            entry=preflight['splits'][split]; source=base/'windows'/split/'manifest.json'
            if sha256_file(source)!=entry['window_manifest_sha256']:raise ValueError(f'Changed manifest: {split}')
            raw=json.loads(source.read_text())
            ids=discover_shards(base,split)
            if ids!=list(range(len(entry['shards']))) or raw['num_shards']!=len(ids):raise ValueError('Shard set mismatch')
            if sorted(p.name for p in (base/'windows'/split).glob('shard_*.npz'))!=[f'shard_{i:06d}.npz' for i in ids]:raise ValueError('Window shard set mismatch')
            if raw['split']!=split or raw['num_samples']!=entry['sample_count'] or (raw['num_devices'],raw['window_size'])!=(dims['N'],dims['W']):raise ValueError('Window metadata mismatch')
            timestamps=read_timestamps(base,split,entry)
            ticks=timestamps.astype('datetime64[ns]').astype(np.int64)
            if np.isnat(timestamps).any() or not np.all(np.diff(ticks)==step):
                raise ValueError('Qualitative protocol requires the unchanged regular reference timelines')
            if index_hash(ticks)!=reference['block_plans'][condition]['timestamp_sha256']:raise ValueError('Timestamps changed since primary')
            all_indices=frozen_primary_indices(plans[condition],timestamps,step_ns=step,seed=seed+len(manifests),window_size=dims['W'])
            if index_hash(all_indices)!=reference['primary_index_sha256'][condition]:raise ValueError('Primary index mismatch')
            selected[condition]=select_windows(plans[condition])
            selected_times[condition]=ticks[selected[condition]]
            manifests.append(entry)
        np.savez(output/'selected_windows.npz',**selected,**{c+'_timestamp_ns':t for c,t in selected_times.items()})
        report['selected_index_sha256']={c:index_hash(v) for c,v in selected.items()}
        report['condition_order']=list(CONDITIONS);report['dimensions']=dims
        write_report(path,report)
        colors=['#0072B2','#D55E00','#009E73']
        for level in ('X',)+LEVELS:
            started_level=time.perf_counter();arrays=[]
            for ci,(condition,split) in enumerate(CONDITIONS.items()):
                ids=selected[condition]
                if level=='X':
                    x=np.empty((len(ids),dims['W'],dims['N']),dtype=np.float32);assigned=np.zeros(len(ids),bool)
                    for item in manifests[ci]['shards']:
                        start,end,sid=item['start_index'],item['end_index'],item['shard_index']
                        pos=np.flatnonzero((ids>=start)&(ids<end))
                        if not len(pos):continue
                        with np.load(base/'windows'/split/f'shard_{sid:06d}.npz',allow_pickle=False) as archive:
                            for field,value in dict(split=split,shard_index=sid,start_index=start,end_index=end).items():
                                if archive[field].item()!=value:raise ValueError('Window metadata changed')
                            rows=archive['X']
                            if rows.shape!=(end-start,dims['W'],dims['N']) or rows.dtype!=np.float32 or not np.isfinite(rows).all():raise ValueError('Invalid input tensor')
                            x[pos]=rows[ids[pos]-start];assigned[pos]=True
                    if not assigned.all():raise ValueError('Missing input rows')
                else:x=collect_level(base,split,level,ids,manifests[ci],checkpoint_hash,dimensions)
                arrays.append(x)
            rows=np.concatenate(arrays);projection,summary=pooled_pca(rows)
            np.savez(output/f'view_{level}.npz',rows=rows,**projection)
            report['levels'][level]=dict(summary=summary,elapsed_seconds=time.perf_counter()-started_level)
            scores=projection['scores'];fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
            for ci,condition in enumerate(CONDITIONS):
                group=scores[ci*128:(ci+1)*128];times=selected_times[condition].astype('datetime64[ns]')
                axes[0].scatter(group[:,0],group[:,1],s=12,alpha=.6,color=colors[ci],label=condition)
                axes[1].scatter(times,group[:,0],s=12,alpha=.6,color=colors[ci])
            axes[0].set(xlabel=f"PC1 ({summary['pc1_variance_fraction']:.1%} variance)",ylabel=f"PC2 ({summary['pc2_variance_fraction']:.1%} variance)")
            axes[0].legend();axes[1].set(xlabel='Recorded date',ylabel='PC1 score');axes[1].tick_params(axis='x',rotation=30)
            fig.suptitle(f'{level}: fixed 384-window subset; centered pooled PCA, descriptive only')
            fig.savefig(output/f'pca_{level}.png',dpi=180);fig.savefig(output/f'pca_{level}.pdf');plt.close(fig)
            write_report(path,report);print(f'{level}: PCA saved; {time.perf_counter()-started_level:.1f}s',flush=True)
        report['status']='COMPLETE';report['scientific_claim_decision']='not_performed'
    except BaseException as exc:
        report['status']='INTERRUPTED' if isinstance(exc,KeyboardInterrupt) else 'FAILED'
        report['error']=f'{type(exc).__name__}: {exc}'
        raise
    finally:
        report['elapsed_seconds']=time.perf_counter()-started;report['peak_rss_mib']=peak_rss_mib()
        write_report(path,report)
        print(f"Qualitative {report['status']}: {path}",flush=True)


if __name__=='__main__':main()
