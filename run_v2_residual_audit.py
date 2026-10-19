#!/usr/bin/env python3
"""Read-only residual decomposition and training-target mean baseline."""
import argparse
from datetime import datetime, timezone
import inspect
import json
from pathlib import Path
import platform
import time
import uuid
import numpy as np
import yaml
from src.utils.get_folders_utils import get_processed_folder
from src.scientific_validation.v2 import preflight as preflight_helper
from src.scientific_validation.v2 import forecast_diagnostic as helper
from src.scientific_validation.v2 import hbf_pilot as pilot
from src.scientific_validation.v2 import residual_audit as residual_helper
from src.scientific_validation.v2.residual_audit import PREDICTORS, fit_train_mean, decompose, compare_to_baseline
from src.scientific_validation.v2.preflight import require, sha256_file, config_differences, step_ns
from src.scientific_validation.v2.forecast_diagnostic import (
    LEVELS, BLOCKS, ROWS, BATCH, SEEDS, write_json, select_blocks,
    block_shards, read_block,
)

PINNED_CHECKPOINT = '4b036348b014652bb84e0f5b44a2b46374bae00213f4ad07b1d4746ff862157b'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project_root_dir')
    parser.add_argument('--pilot_report', type=Path, required=True)
    parser.add_argument('--forecast_report', type=Path, help='Optional relocated original diagnostic')
    parser.add_argument('--preflight_report', type=Path, help='Optional relocated linked preflight')
    parser.add_argument('--checkpoint', type=Path, help='Optional relocated copy of the SAME checkpoint')
    parser.add_argument('--device', choices=('cpu', 'cuda'), help='Default: CUDA when available')
    args = parser.parse_args()
    with open('configs/config.yaml', encoding='utf-8') as stream:
        config = yaml.safe_load(stream)
    require(isinstance(config, dict), 'Configuration must be a mapping')
    if args.project_root_dir is not None:
        config['project_root_dir'] = args.project_root_dir
    base = Path(get_processed_folder(config)).resolve()
    root = Path(__file__).resolve().parent
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8]
    output = base/'v2_validation'/'residual_audit'/run_id
    output.mkdir(parents=True, exist_ok=False)
    report_path = output/'residual_report.json'
    started = time.perf_counter()
    report = dict(status='RUNNING', run_id=run_id, configuration=config,
                  protocol='v2_residual_decomposition_v1', optimizer_steps=0,
                  levels=list(LEVELS), splits={}, source_code_sha256={},
                  budget=dict(total_paired_windows=4096, windows_per_split=2048,
                              blocks_per_split=BLOCKS, windows_per_block=ROWS, batch_size=BATCH),
                  limitations=[
                      'Exploratory read-only follow-up; no optimization or checkpoint selection.',
                      'Train-mean vectors are fitted only to 2048 selected training targets, not the whole training split.',
                      'Training baseline evaluation is in-sample; validation has already been inspected.',
                      'Centering within an evaluated split/block is descriptive decomposition, not a deployable bias correction.',
                      'Centered alignment pools coordinate covariances; it can conceal coordinate/block heterogeneity.',
                      'Adjacent windows overlap; no confidence intervals, p-values or independence claim.',
                      'Only original and saved final-step HBF weights are available; no reconstruction of intermediate pilot states.',
                      'Positive covariance alone is not evidence of prediction beyond persistence.'])
    write_json(report_path, report)
    print(f'Diagnostic report: {report_path}', flush=True)
    tracked = {}

    def pin(path, expected=None):
        path = Path(path).resolve()
        digest = sha256_file(path)
        require(expected is None or digest == expected, f'Hash mismatch: {path}')
        require(str(path) not in tracked or tracked[str(path)] == digest, f'Input changed: {path}')
        tracked[str(path)] = digest
        return digest

    try:
        pilot_hash = pin(args.pilot_report)
        completed = json.loads(args.pilot_report.read_text())
        require(completed['status'] == 'COMPLETE' and completed['protocol'] == 'v2_hbf_only_640_v1', 'Expected completed HBF pilot')
        require(completed['optimizer_steps'] == 640, 'Expected fixed final pilot step 640')
        require(completed['checkpoint']['sha256'] == PINNED_CHECKPOINT, 'Pilot parent checkpoint mismatch')
        for name, digest in completed['artifacts'].items():
            require(Path(name).name == name, 'Invalid pilot artifact name')
            pin(args.pilot_report.parent/name, digest)
        for old, digest in completed['source_code_sha256'].items():
            source = root/'src'/old.split('/src/',1)[1] if '/src/' in old else root/Path(old).name
            pin(source,digest)
        report['pilot_reference'] = dict(path=str(args.pilot_report.resolve()), sha256=pilot_hash, run_id=completed['run_id'])
        args.forecast_report = args.forecast_report or Path(completed['forecast_reference']['path'])
        parent_hash = pin(args.forecast_report, completed['forecast_reference']['sha256'])
        parent = json.loads(args.forecast_report.read_text())
        require(parent['status'] == 'COMPLETE' and parent['protocol'] == 'v2_forecast_32x64_v1', 'Expected completed forecasting diagnostic')
        require(parent['checkpoint']['sha256'] == PINNED_CHECKPOINT, 'Unexpected parent checkpoint')
        for name in ('selection_plan.json', 'selected_windows.npz', 'per_window_errors.npz', 'block_statistics.json'):
            pin(args.forecast_report.parent/name, parent['artifacts'][name])
        report['forecast_reference'] = dict(path=str(args.forecast_report.resolve()), sha256=parent_hash, run_id=parent['run_id'])
        args.preflight_report = args.preflight_report or Path(parent['preflight']['path'])
        preflight_hash = pin(args.preflight_report, parent['preflight']['sha256'])
        for old, digest in parent['source_code_sha256'].items():
            path = root/'src'/old.split('/src/', 1)[1] if '/src/' in old else root/Path(old).name
            pin(path, digest)
        preflight = json.loads(args.preflight_report.read_text())
        require(preflight.get('status') == 'COMPLETE', 'Preflight is not COMPLETE')
        require(preflight['checkpoint']['sha256'] == PINNED_CHECKPOINT, 'Unexpected preflight checkpoint')
        require(preflight['checkpoint']['strict_model_load'] == 'PASS', 'Preflight model load did not pass')
        require(preflight['semantics']['fixed_buffer_comparison'] == 'PASS', 'Preflight buffers did not pass')
        require(preflight['split_boundaries']['status'] == 'PASS', 'Preflight split boundaries did not pass')
        differences = config_differences(config, preflight['configuration'])
        require(all(d['category'] == 'path_only' for d in differences),
                'Configuration changed since preflight; rerun preflight with the active configuration')
        report['configuration_differences_from_preflight'] = differences
        report['preflight'] = dict(path=str(args.preflight_report.resolve()), sha256=preflight_hash,
                                   run_id=preflight['run_id'], mode=preflight.get('mode'))
        checkpoint = args.checkpoint or Path(config['hdeg']['hbf']['checkpoint'])
        if not checkpoint.is_absolute():
            checkpoint = base/'hdeg_checkpoints'/checkpoint
        checkpoint_hash = pin(checkpoint, PINNED_CHECKPOINT)
        report['checkpoint'] = dict(path=str(checkpoint.resolve()), sha256=checkpoint_hash)
        # Relocate source paths by project-relative suffix, never by guessed remote roots.
        for old, expected in preflight['source_code_sha256'].items():
            if old == 'v2/preflight.py':
                source = Path(preflight_helper.__file__)
            elif '/src/' in old:
                source = root/'src'/old.split('/src/', 1)[1]
            else:
                source = root/Path(old).name
            report['source_code_sha256'][str(source.resolve())] = pin(source, expected)
        for source in (__file__, helper.__file__, pilot.__file__, residual_helper.__file__):
            report['source_code_sha256'][str(Path(source).resolve())] = pin(source)
        pin(Path('configs/config.yaml'))
        semantics_path = Path(config['project_root_dir'])/'configs'/'hdeg'/'behavioral_states.yaml'
        pin(semantics_path, preflight['semantics']['behavioral_states_sha256'])
        pin(base/'devices.json', preflight['semantics']['devices_sha256'])
        w, n = preflight['dimensions']['W'], preflight['dimensions']['N']
        step = step_ns(config['preprocessing']['downsample_freq'])
        from run_dbrl_sharded import load_manifest, resolve_shard_path
        plans, sources, selected_arrays = {}, {}, {}
        for split in SEEDS:
            previous = preflight['splits'][split]
            require(previous['status'] == 'PASS', f'{split} preflight did not pass')
            pin(base/'windows'/split/'manifest.json', previous['manifest_sha256'])
            manifest = load_manifest(windows_dir=base/'windows', split=split)
            require(manifest['num_samples'] == previous['sample_count'] and
                    manifest['window_size'] == w and manifest['num_devices'] == n,
                    'Manifest dimensions changed')
            blocks = select_blocks(manifest['num_samples'], w, SEEDS[split])
            shards = []
            require(len(manifest['shards']) == len(previous['shards']), 'Shard count changed')
            for entry, old in zip(manifest['shards'], previous['shards']):
                source = Path(resolve_shard_path(base/'windows'/split, entry)).resolve()
                require(source.parent == (base/'windows'/split).resolve(), 'Shard outside split directory')
                require(source.name == Path(old['file']).name, 'Shard order changed')
                shards.append(dict(old, file=str(source)))
            chosen = {s['shard_index'] for b in blocks for s in block_shards(b, shards)}
            print(f'{split}: pinning {len(chosen)} selected source shards...', flush=True)
            for s in shards:
                if s['shard_index'] in chosen:
                    s['sha256'] = pin(s['file'], s.get('sha256'))
            sources[split] = [s for s in shards if s['shard_index'] in chosen]
            indices = np.concatenate([np.arange(b['start_index'], b['end_index'], dtype=np.int64) for b in blocks])
            selected_arrays[split+'_indices'] = indices
            selected_arrays[split+'_block'] = np.repeat(np.arange(BLOCKS), ROWS)
            selected_arrays[split+'_start_ns'] = previous['first_input_start_ns']+indices*step
            selected_arrays[split+'_target_end_ns'] = selected_arrays[split+'_start_ns']+w*step
            plans[split] = dict(seed=SEEDS[split], blocks=blocks, shards=sources[split],
                                timestamp_basis='Expected from preflight; verified against selected rows during loading')
        with np.load(args.forecast_report.parent/'selected_windows.npz', allow_pickle=False) as original:
            require(set(original.files) == set(selected_arrays), 'Selection fields mismatch')
            for key, value in selected_arrays.items():
                require(np.array_equal(original[key], value), f'Frozen selection differs: {key}')
        original_plan = json.loads((args.forecast_report.parent/'selection_plan.json').read_text())
        for split in SEEDS:
            original_hashes = {s['shard_index']: s['sha256'] for s in original_plan['splits'][split]['shards']}
            for shard in sources[split]:
                require(shard['sha256'] == original_hashes[shard['shard_index']], 'Source changed since forecasting diagnostic')
        np.savez_compressed(output/'selected_windows.npz', **selected_arrays)
        write_json(output/'selection_plan.json', dict(protocol=report['protocol'], splits=plans,
                   checkpoint_sha256=checkpoint_hash, preflight_sha256=preflight_hash,
                   selection_sha256=pin(output/'selected_windows.npz'),
                   boundary_rule='Entire [start,start+64+W) observation span lies within its chronological stratum',
                   rng='NumPy Generator(PCG64); fixed seeds; no redraws or outcome-based selection'))
        pin(output/'selection_plan.json')
        print('Selection verified. Loading original model for read-only decomposition.', flush=True)
        import torch
        from src.training.train_hdeg_e2e import build_live_model
        from src.common.graph.semantics import load_behavioral_state_config
        device = torch.device(args.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.manual_seed(42)
        np.random.seed(42)
        report['runtime'] = dict(python=platform.python_version(), numpy=np.__version__,
                                torch=torch.__version__, device=str(device),
                                inference_dtype='float32', reduction_dtype='float64', tf32=False,
                                bitwise_cross_platform_reproducibility='not_guaranteed')
        payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
        state = payload['model_state_dict']
        require(not any(d['category'] == 'experiment_critical' for d in
                        config_differences(config, payload['config'])), 'Checkpoint configuration mismatch')
        semantics = load_behavioral_state_config(config_path=semantics_path, devices_path=base/'devices.json')
        model = build_live_model(config=config, behavioral_config=semantics,
                                 compatibility_mask=semantics.torch_mask(dtype=torch.float32,
                                     device=torch.device('cpu'), clone=True), device=torch.device('cpu'))
        # Verify the imported builder and modules really came from the pinned files.
        for obj in (build_live_model, load_behavioral_state_config, load_manifest,
                    get_processed_folder, *(type(m) for m in model.children())):
            source = str(Path(inspect.getsourcefile(obj)).resolve())
            require(source in tracked, f'Imported source was not pinned: {source}')
            pin(source, tracked[source])
        for name, value in model.named_buffers():
            require(name in state and value.dtype == state[name].dtype and torch.equal(value, state[name]),
                    f'Fixed buffer mismatch: {name}')
        expected = model.state_dict()
        require(set(expected) == set(state), 'Checkpoint model keys mismatch')
        for name, value in expected.items():
            require(value.shape == state[name].shape and value.dtype == state[name].dtype and
                    torch.isfinite(state[name]).all().item(), f'Invalid checkpoint tensor: {name}')
        model.load_state_dict(state, strict=True)
        model.to(device).eval()
        model.requires_grad_(False)
        for obj in (*(type(m) for m in model.modules()), type(model.dbrl.graph_builder)):
            source = inspect.getsourcefile(obj)
            if source and Path(source).resolve().is_relative_to(root):
                report['source_code_sha256'][str(Path(source).resolve())] = pin(source)
        report['checkpoint'].update(epoch=payload['epoch'], strict_load='PASS')
        report['dimensions'] = preflight['dimensions']
        report['forward_calls'] = dict(encoder=0, forecaster=0)
        del payload, state, expected
        shapes = dict(Z=(n, preflight['dimensions']['D']),
                      S=(preflight['dimensions']['K'], preflight['dimensions']['D']),
                      S_tilde=(preflight['dimensions']['K'], preflight['dimensions']['D']),
                      g=(preflight['dimensions']['D'],))
        frozen_before = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()
                         if not name.startswith('hbf.')}
        initial_hbf = {name: value.detach().cpu().clone() for name, value in model.hbf.state_dict().items()}
        write_json(output/'residual_plan.json', dict(protocol=report['protocol'], predictors=list(PREDICTORS),
                   training_mean_fit_split='train', training_mean_fit_rows=2048, optimizer_steps=0,
                   original_checkpoint_sha256=checkpoint_hash,
                   pilot_final_sha256=completed['artifacts']['hbf_final.pt'],
                   selection_sha256=sha256_file(output/'selected_windows.npz')))
        pin(output/'residual_plan.json')
        cache = {}
        for split in SEEDS:
            parts = {role:{level:[] for level in LEVELS} for role in ('current', 'target')}
            for block in plans[split]['blocks']:
                x, y = read_block(block, sources[split], split, w, n,
                                  preflight['splits'][split]['first_input_start_ns'], step)
                with torch.no_grad():
                    for start in range(0, ROWS, BATCH):
                        for role, data in (('current', x), ('target', y)):
                            values = model.encode_window(torch.from_numpy(data[start:start+BATCH]).to(device))
                            report['forward_calls']['encoder'] += 1
                            for level in LEVELS:
                                a = values[level]
                                require(tuple(a.shape)==(BATCH,*shapes[level]) and a.dtype==torch.float32 and
                                        torch.isfinite(a).all().item(), f'Invalid encoding: {level}')
                                parts[role][level].append(a.detach().cpu().clone())
                print(f'{split}: cached block {block["block"]+1}/32', flush=True)
            cache[split] = {role:{level:torch.cat(arrays) for level,arrays in levels.items()}
                            for role,levels in parts.items()}
            del parts
        cache_hash = pilot.cache_digest(cache)
        report['representation_cache'] = dict(storage='RAM only', sha256=cache_hash,
            bytes=sum(v.numel()*v.element_size() for s in cache.values() for r in s.values() for v in r.values()))
        train_means = {level:fit_train_mean(cache['train']['target'][level].numpy()) for level in LEVELS}
        np.savez_compressed(output/'train_mean_vectors.npz', **train_means)
        report['train_mean_fit'] = dict(split='train', rows=2048, dtype='float64',
            source='Frozen encoder applied to selected training target windows only')
        predictions = {}
        replay = {}
        for predictor in ('original_hbf', 'pilot_hbf'):
            if predictor == 'pilot_hbf':
                final_path = args.pilot_report.parent/'hbf_final.pt'
                final = torch.load(final_path, map_location='cpu', weights_only=False)
                require(final['artifact_type'] == 'hbf_only_pilot_final' and final['optimizer_steps'] == 640 and
                        final['parent_checkpoint_sha256'] == PINNED_CHECKPOINT and final['protocol'] == completed['protocol'],
                        'Invalid final HBF artifact lineage')
                require(final['selection_sha256'] == completed['artifacts']['selected_windows.npz'], 'Final HBF selection mismatch')
                fs = final['hbf_state_dict']
                expected = model.hbf.state_dict()
                require(set(fs) == set(expected), 'Final HBF keys mismatch')
                for name, value in expected.items():
                    require(value.shape == fs[name].shape and value.dtype == fs[name].dtype and torch.isfinite(fs[name]).all().item(),
                            f'Invalid final HBF tensor: {name}')
                model.hbf.load_state_dict(fs, strict=True)
                hbf_expected = {k:v.clone() for k,v in fs.items()}
                del final, fs, expected
            else:
                hbf_expected = initial_hbf
            model.eval()
            predictions[predictor] = {}
            for split in SEEDS:
                values = {level:[] for level in LEVELS}
                with torch.no_grad():
                    for start in range(0,2048,32):
                        current = {level:t[start:start+32].to(device) for level,t in cache[split]['current'].items()}
                        out = model.forecast(current)
                        report['forward_calls']['forecaster'] += 1
                        for level in LEVELS:
                            a = out[level].detach().cpu().numpy().copy()
                            require(a.shape==(32,*shapes[level]) and np.isfinite(a).all(), 'Invalid forecast')
                            values[level].append(a)
                predictions[predictor][split] = {level:np.concatenate(arrays) for level,arrays in values.items()}
            for name,value in model.hbf.state_dict().items():
                require(torch.equal(value.detach().cpu(),hbf_expected[name]), f'HBF mutated during read-only evaluation: {name}')
            print(f'Computed {predictor} on both fixed splits', flush=True)
        all_errors, saved_moments, block_statistics = {}, {}, {}
        for split in SEEDS:
            result, blocks = {}, {predictor:{} for predictor in PREDICTORS}
            errors = {predictor:[] for predictor in PREDICTORS}
            centered_errors = {predictor:[] for predictor in PREDICTORS}
            for level in LEVELS:
                t = cache[split]['target'][level].numpy()
                level_result = {}
                for predictor in PREDICTORS:
                    if predictor == 'train_mean':
                        prediction = np.broadcast_to(train_means[level], (2048, train_means[level].size)).reshape(t.shape)
                    elif predictor == 'persistence':
                        prediction = cache[split]['current'][level].numpy()
                    else:
                        prediction = predictions[predictor][split][level]
                    summary, e, ce, moments = decompose(prediction,t)
                    errors[predictor].append(e)
                    centered_errors[predictor].append(ce)
                    level_result[predictor] = summary
                    for field,value in moments.items():
                        saved_moments[f'{split}__{predictor}__{level}__{field}'] = value
                    if split == 'train' and predictor == 'train_mean':
                        require(summary['mean_offset_mse'] == 0, 'Train mean has nonzero in-sample offset')
                    per_block = []
                    for bid in range(32):
                        sl = slice(bid*64,(bid+1)*64)
                        block,_,_,_ = decompose(prediction[sl], t[sl])
                        per_block.append(dict(block_id=bid, **block))
                    blocks[predictor][level] = per_block
                result[level] = level_result
            for predictor in PREDICTORS:
                matrix = np.stack(errors[predictor],axis=1)
                all_errors[f'{split}__{predictor}__mse'] = matrix
                all_errors[f'{split}__{predictor}__centered_mse'] = np.stack(centered_errors[predictor],axis=1)
                for j,level in enumerate(LEVELS):
                    result[level][predictor]['comparisons'] = {
                        baseline:compare_to_baseline(matrix[:,j],errors[baseline][j])
                        for baseline in ('persistence','train_mean')}
            report['splits'][split] = dict(paired_windows=2048, levels=result)
            block_statistics[split] = blocks
        with np.load(args.forecast_report.parent/'per_window_errors.npz',allow_pickle=False) as original, \
             np.load(args.pilot_report.parent/'per_window_errors.npz',allow_pickle=False) as pilot_errors:
            for split in SEEDS:
                for predictor, reference in (
                    ('original_hbf',original[split+'_hbf_mse']),
                    ('pilot_hbf',pilot_errors['step_0640_'+split+'_hbf_mse']),
                    ('persistence',original[split+'_persistence_mse'])):
                    actual = all_errors[f'{split}__{predictor}__mse']
                    good = np.allclose(actual,reference,rtol=1e-3,atol=1e-14)
                    replay[split+'__'+predictor] = dict(within_tolerance=bool(good),
                        max_absolute_difference=float(np.max(np.abs(actual-reference))))
                    require(good, f'Forecast replay mismatch: {split}/{predictor}')
        report['replay_checks'] = dict(rtol=1e-3,atol=1e-14,arrays=replay)
        np.savez_compressed(output/'per_window_errors.npz',levels=np.array(LEVELS), **all_errors)
        np.savez_compressed(output/'coordinate_moments.npz', **saved_moments)
        write_json(output/'block_decomposition.json',block_statistics)
        for name,value in model.state_dict().items():
            if not name.startswith('hbf.'):
                require(torch.equal(value.detach().cpu(),frozen_before[name]),f'Frozen encoder/MO state changed: {name}')
        require(all(p.grad is None and not p.requires_grad for p in model.parameters()), 'Unexpected gradient state')
        require(pilot.cache_digest(cache)==cache_hash,'Representation cache changed')
        require(report['forward_calls']==dict(encoder=256,forecaster=256), 'Incomplete read-only forward budget')
        for path,digest in tracked.items():
            require(sha256_file(path)==digest,f'Input changed: {path}')
        report['checks'] = dict(encoder_and_mo_unchanged=True, each_hbf_unchanged_during_its_evaluation=True,
            input_files_unchanged=True, gradient_buffers_absent=True, representation_cache_unchanged=True)
        report['input_sha256'] = tracked
        report['artifacts'] = {name:sha256_file(output/name) for name in ('selection_plan.json','selected_windows.npz',
            'residual_plan.json','train_mean_vectors.npz','per_window_errors.npz','coordinate_moments.npz','block_decomposition.json')}
        report['scientific_claim_decision'] = 'descriptive_decomposition_requires_review'
        report['status'] = 'COMPLETE'
    except Exception as exc:
        report.update(status='FAILED',error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        report['elapsed_seconds'] = time.perf_counter()-started
        write_json(report_path,report)
    print(f'COMPLETE: {report_path}',flush=True)


if __name__ == '__main__':
    main()
