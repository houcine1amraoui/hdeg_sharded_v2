#!/usr/bin/env python3
"""Standalone R1: sample local pairs before computing distances."""
import argparse
import copy
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
from src.scientific_validation.v1.v1b_metrics import percentile_summary
from src.scientific_validation.v1.r3_pilot import CONDITIONS, LEVELS, positive_int, index_hash
from src.scientific_validation.v1.r3_sensitivity import validate_primary
from src.scientific_validation.v1 import r1_standalone as helper
from src.scientific_validation.v1.r1_standalone import chunk_population, sample_pairs
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
    output=base/'v1_validation'/'r1_standalone'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8])
    output.mkdir(parents=True)
    started=time.perf_counter()
    report=dict(status='RUNNING',dataset=base.name,configuration=config,levels={},
        measurement_family='within_level_representation_geometry',
        pair_population='Within each chunk within each shard; pooled val, actor2_test, actor1_test',
        sampling_method='uniform_without_replacement_from_pair_ranks_before_distance_computation',
        legacy_equivalence='Same eligible pair population and uniform-subset distribution as the reservoir; not the same realized sample or RNG sequence.',
        limitation='Local chunk geometry, not global geometry, semantic coherence, or cross-level superiority. No confidence intervals or scientific pass/fail.',
        memory_measure='Process lifetime peak RSS, including imports and loaded shards')
    path=output/'r1_report.json'
    write_report(path,report)
    print(f'R1 output: {output}',flush=True)
    try:
        settings=config['v1_validation']
        capacity=positive_int(settings['r1-reservoir'],'r1-reservoir')
        chunk_size=positive_int(settings['r1-chunk'],'r1-chunk')
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
            settings=dict(reservoir_capacity=capacity,chunk_size=chunk_size,seed=seed,pair_batch_size=1024))
        # Record every directly used project module; preserve the inherited I/O helper.
        import run_r3_pilot as pilot
        if sha256_file(Path(pilot.__file__))!=reference['source_code_sha256']['run_r3_pilot.py']:
            raise ValueError('Pilot I/O helper changed since primary')
        report['source_code_sha256']={str(Path(__file__).name):sha256_file(Path(__file__)),
                                     'r1_standalone.py':sha256_file(Path(helper.__file__))}
        import inspect
        for fn in (collect_level,discover_shards,sha256_file,percentile_summary,get_processed_folder,validate_primary):
            source=Path(inspect.getsourcefile(fn))
            report['source_code_sha256'][str(source)]=sha256_file(source)
        manifests=[]; step=sampling_step_ns(config['preprocessing']['downsample_freq'])
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
                raise ValueError('R1 local-chunk protocol requires regular timestamps; review gaps before changing pair eligibility')
            if index_hash(ticks)!=reference['block_plans'][condition]['timestamp_sha256']:raise ValueError('Timestamps changed since primary')
            manifests.append(entry)
        chunks=chunk_population(manifests,chunk_size)
        samples={};ranks={};total=0
        for offset,level in enumerate(LEVELS):
            samples[level],ranks[level],total=sample_pairs(chunks,capacity,seed+offset)
        np.savez(output/'selected_pairs.npz',**samples)
        np.savez(output/'selected_pair_ranks.npz',**ranks)
        report['pair_plan']=dict(condition_order=list(CONDITIONS),pair_columns=['condition_index','shard_index','window_i','window_j'],
            chunks=[list(c) for c in chunks],chunk_columns=['condition_index','shard_index','start_index','rows','pairs'],
            total_eligible_pairs=total,selected_pairs_per_level=min(capacity,total),
            pair_sha256={l:index_hash(p) for l,p in samples.items()},level_seeds={l:seed+i for i,l in enumerate(LEVELS)})
        report['compute_budget']=dict(legacy_distance_evaluations=total*4,standalone_distance_evaluations=min(capacity,total)*4,
            reduction_factor=total/min(capacity,total))
        report['dimensions']=dims
        write_report(path,report)
        print(f'Eligible pairs/level: {total:,}; selected: {min(capacity,total):,}; distance-work reduction: {total/min(capacity,total):.1f}x',flush=True)
        for level in LEVELS:
            pairs=samples[level]; distances=np.empty(len(pairs)); t=time.perf_counter()
            groups=np.unique(pairs[:,:2],axis=0)
            for group,(condition,sid) in enumerate(groups,1):
                positions=np.flatnonzero((pairs[:,0]==condition)&(pairs[:,1]==sid))
                chosen=pairs[positions,2:];indices=np.unique(chosen)
                split=list(CONDITIONS.values())[condition]
                rows=collect_level(base,split,level,indices,manifests[condition],checkpoint_hash,dimensions)
                local=np.searchsorted(indices,chosen)
                for start in range(0,len(local),1024):
                    stop=min(start+1024,len(local)); ii,jj=local[start:stop].T
                    a=rows[ii].reshape(len(ii),-1).astype(np.float64)
                    b=rows[jj].reshape(len(jj),-1).astype(np.float64)
                    distances[positions[start:stop]]=np.linalg.norm(a-b,axis=1)
                del rows
                if group%10==0 or group==len(groups):print(f'{level}: shard group {group}/{len(groups)}; elapsed {time.perf_counter()-t:.1f}s',flush=True)
            if not np.isfinite(distances).all():raise ValueError('Nonfinite pair distances')
            np.save(output/f'distances_{level}.npy',distances)
            report['levels'][level]=dict(summary=percentile_summary(distances),elapsed_seconds=time.perf_counter()-t,
                sampled_pairs_by_condition={c:int(np.sum(pairs[:,0]==i)) for i,c in enumerate(CONDITIONS)})
            write_report(path,report)
        report['status']='COMPLETE';report['scientific_claim_decision']='not_performed'
    except BaseException as exc:
        report['status']='INTERRUPTED' if isinstance(exc,KeyboardInterrupt) else 'FAILED'
        report['error']=f'{type(exc).__name__}: {exc}'
        raise
    finally:
        report['elapsed_seconds']=time.perf_counter()-started;report['peak_rss_mib']=peak_rss_mib()
        write_report(path,report)
        print(f"R1 {report['status']}: {path}",flush=True)


if __name__=='__main__':main()
