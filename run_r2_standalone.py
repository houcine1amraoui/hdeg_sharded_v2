#!/usr/bin/env python3
"""Standalone R2: input/representation distance relationships with stratum diagnostics."""
import argparse
import tempfile
import shutil
import gc
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
from src.scientific_validation.v1.v1b_sampling import deterministic_pairs
from src.scientific_validation.v1.r3_pilot import CONDITIONS, LEVELS, positive_int, index_hash
from src.scientific_validation.v1.r3_sensitivity import validate_primary
from src.scientific_validation.v1 import r2_standalone as helper
from src.scientific_validation.v1.r2_standalone import paired_distances, summarize_relationships
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
    output=base/'v1_validation'/'r2_standalone'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8])
    output.mkdir(parents=True)
    started=time.perf_counter()
    report=dict(status='RUNNING',dataset=base.name,configuration=config,levels={},
        measurement_family='behavioral_relationship_preservation',
        pair_population='All unordered pairs from concatenated val, actor2_test, actor1_test; no self-pairs',
        sampling_method='existing deterministic_pairs; uniform without replacement; shared across all levels',
        primary_measure='Pooled Spearman correlation using average ranks for ties',
        behavior_distance='Frobenius(X_i-X_j)/(W*N); existing preprocessed input scale',
        representation_distance='Within-native-level Euclidean/Frobenius distance',
        limitation='Pooled correlation may reflect between-condition differences. Six condition-pair correlations are descriptive diagnostics, not adjusted effects. Shared endpoints and temporal dependence invalidate naive independent-pair inference. No p-values, CIs, or scientific pass/fail.',
        memory_measure='Process lifetime peak RSS, including imports and loaded shards')
    path=output/'r2_report.json'
    write_report(path,report)
    print(f'R2 output: {output}',flush=True)
    try:
        settings=config['v1_validation']
        count=positive_int(settings['r2-pairs'],'r2-pairs')
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
            settings=dict(requested_pair_count=count,seed=seed,pair_batch_size=1024))
        # Record every directly used project module; preserve the inherited I/O helper.
        import run_r3_pilot as pilot
        if sha256_file(Path(pilot.__file__))!=reference['source_code_sha256']['run_r3_pilot.py']:
            raise ValueError('Pilot I/O helper changed since primary')
        report['source_code_sha256']={str(Path(__file__).name):sha256_file(Path(__file__)),
                                     'r2_standalone.py':sha256_file(Path(helper.__file__))}
        import inspect
        for fn in (collect_level,discover_shards,sha256_file,deterministic_pairs,get_processed_folder,validate_primary):
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
                raise ValueError('R2 requires the reference regular timelines; review changed timestamps before execution')
            if index_hash(ticks)!=reference['block_plans'][condition]['timestamp_sha256']:raise ValueError('Timestamps changed since primary')
            manifests.append(entry)
        sizes=np.array([m['sample_count'] for m in manifests],dtype=np.int64)
        ends=np.cumsum(sizes);offsets=np.r_[0,ends[:-1]]
        pairs=deterministic_pairs(int(ends[-1]),count,seed)
        if len(pairs)<2:raise ValueError('R2 requires at least two sampled pairs')
        endpoints=np.unique(pairs);local_pairs=np.searchsorted(endpoints,pairs)
        condition_pairs=np.searchsorted(ends,pairs,side='right')
        endpoint_conditions=np.searchsorted(ends,endpoints,side='right')
        np.savez(output/'selected_pairs.npz',combined_indices=pairs,condition_indices=condition_pairs,
                 split_local_indices=pairs-offsets[condition_pairs],unique_endpoints=endpoints)
        report['pair_plan']=dict(condition_order=list(CONDITIONS),split_order=list(CONDITIONS.values()),
            split_sizes=sizes.tolist(),split_offsets=offsets.tolist(),total_windows=int(ends[-1]),
            pair_count=len(pairs),unique_endpoint_count=len(endpoints),
            pair_sha256=index_hash(pairs),endpoint_sha256=index_hash(endpoints),
            sampling_shared_across_levels=True)
        report['dimensions']=dims
        report['compute_budget']=dict(input_distance_evaluations=len(pairs),representation_distance_evaluations=4*len(pairs),
            largest_endpoint_cache_bytes=len(endpoints)*4*max(dims['W']*dims['N'],dims['N']*dims['D'],dims['K']*dims['D'],dims['D']))
        write_report(path,report)
        print(f'R2 pairs: {len(pairs):,}; unique endpoints: {len(endpoints):,}',flush=True)
        # A local disk cache holds one modality at a time; never retain all levels.
        with tempfile.TemporaryDirectory(prefix='hdeg_r2_') as temporary:
            report['cache_device']='local temporary directory; one modality at a time'
            for level in ('X',)+LEVELS:
                t=time.perf_counter()
                tail={'X':(dims['W'],dims['N']),'Z':(dims['N'],dims['D']),
                      'S':(dims['K'],dims['D']),'S_tilde':(dims['K'],dims['D']),'g':(dims['D'],)}[level]
                required=len(endpoints)*int(np.prod(tail))*4
                if shutil.disk_usage(temporary).free < required+64*1024**2:raise OSError('Insufficient local temporary disk space')
                cache_path=Path(temporary)/'endpoints.npy'
                rows=np.lib.format.open_memmap(cache_path,mode='w+',dtype=np.float32,shape=(len(endpoints),)+tail)
                assigned=np.zeros(len(endpoints),dtype=bool)
                completed=0
                for ci,split in enumerate(CONDITIONS.values()):
                    split_local=endpoints-offsets[ci]
                    for item in manifests[ci]['shards']:
                        start,end,sid=item['start_index'],item['end_index'],item['shard_index']
                        positions=np.flatnonzero((endpoint_conditions==ci)&(split_local>=start)&(split_local<end))
                        if not len(positions):continue
                        selected=split_local[positions]
                        if level=='X':
                            source=base/'windows'/split/f'shard_{sid:06d}.npz'
                            with np.load(source,allow_pickle=False) as archive:
                                for field,value in dict(split=split,shard_index=sid,start_index=start,end_index=end).items():
                                    if archive[field].item()!=value:raise ValueError(f'Window metadata mismatch: {source}/{field}')
                                x=archive['X']
                                if x.shape!=(end-start,)+tail or x.dtype!=np.float32 or not np.isfinite(x).all():raise ValueError(f'Invalid window tensor: {source}')
                                rows[positions]=x[selected-start]
                                del x
                        else:
                            extracted=collect_level(base,split,level,selected,manifests[ci],checkpoint_hash,dimensions)
                            rows[positions]=extracted
                            del extracted
                        assigned[positions]=True;completed+=1
                        if completed%10==0:print(f'{level}: loaded {completed} shard groups; {time.perf_counter()-t:.1f}s',flush=True)
                if not assigned.all():raise ValueError('Unassigned pair endpoints')
                rows.flush()
                distances=paired_distances(rows,local_pairs,denominator=dims['W']*dims['N'] if level=='X' else 1)
                del rows
                gc.collect();cache_path.unlink()
                np.save(output/f'distances_{level}.npy',distances)
                if level=='X':
                    input_distances=distances
                    report['input_loading_and_distance_seconds']=time.perf_counter()-t
                else:
                    report['levels'][level]=summarize_relationships(input_distances,distances,condition_pairs)
                    report['levels'][level]['elapsed_seconds']=time.perf_counter()-t
                write_report(path,report)
                print(f'{level} complete: {time.perf_counter()-t:.1f}s',flush=True)
        report['status']='COMPLETE';report['scientific_claim_decision']='not_performed'
    except BaseException as exc:
        report['status']='INTERRUPTED' if isinstance(exc,KeyboardInterrupt) else 'FAILED'
        report['error']=f'{type(exc).__name__}: {exc}'
        raise
    finally:
        report['elapsed_seconds']=time.perf_counter()-started;report['peak_rss_mib']=peak_rss_mib()
        write_report(path,report)
        print(f"R2 {report['status']}: {path}",flush=True)


if __name__=='__main__':main()
