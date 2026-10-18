#!/usr/bin/env python3
"""Bounded frozen-model HBF versus persistence diagnostic; no training."""
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
from src.scientific_validation.v2.preflight import require, sha256_file, config_differences, step_ns
from src.scientific_validation.v2.forecast_diagnostic import (
    LEVELS, BLOCKS, ROWS, BATCH, SEEDS, write_json, select_blocks,
    block_shards, read_block, window_mse, error_summary, Moments,
)

PINNED_CHECKPOINT = '4b036348b014652bb84e0f5b44a2b46374bae00213f4ad07b1d4746ff862157b'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project_root_dir')
    parser.add_argument('--preflight_report', type=Path, required=True)
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
    output = base/'v2_validation'/'forecast_diagnostic'/run_id
    output.mkdir(parents=True, exist_ok=False)
    report_path = output/'diagnostic_report.json'
    started = time.perf_counter()
    report = dict(status='RUNNING', run_id=run_id, configuration=config,
                  protocol='v2_forecast_32x64_v1', optimizer_steps=0,
                  levels=list(LEVELS), splits={}, source_code_sha256={},
                  budget=dict(total_paired_windows=4096, windows_per_split=2048,
                              blocks_per_split=BLOCKS, windows_per_block=ROWS, batch_size=BATCH),
                  limitations=[
                      'One frozen epoch; no convergence or model-selection claim.',
                      'Train is in-sample behavior; validation is a bounded held-out diagnostic.',
                      'Adjacent windows overlap heavily; no IID inference, p-values or confidence intervals.',
                      'Targets use the same learned encoder; low errors alone do not establish useful dynamics.',
                      'Variation measures describe scale and exact constancy; no near-collapse threshold is imposed.',
                      'Skill ratios can be unstable for small positive persistence errors; raw denominators are retained.',
                      'No cross-level pooling: representation scales and coordinate counts differ.',
                      'Original training device IDs are absent from this checkpoint; identical-mask permutations remain unverifiable.',
                      'Only selected data are rechecked here; preflight hashed tensor contents in first/middle/last shards only.',
                      'This run neither inherits nor updates Level 1 scientific conclusions.'])
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
        preflight_hash = pin(args.preflight_report)
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
        for source in (__file__, helper.__file__):
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
        np.savez_compressed(output/'selected_windows.npz', **selected_arrays)
        write_json(output/'selection_plan.json', dict(protocol=report['protocol'], splits=plans,
                   checkpoint_sha256=checkpoint_hash, preflight_sha256=preflight_hash,
                   selection_sha256=pin(output/'selected_windows.npz'),
                   boundary_rule='Entire [start,start+64+W) observation span lies within its chronological stratum',
                   rng='NumPy Generator(PCG64); fixed seeds; no redraws or outcome-based selection'))
        pin(output/'selection_plan.json')
        print('Selection frozen. Loading the model; no optimizer will be constructed.', flush=True)
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
        report['checkpoint'].update(epoch=payload['epoch'], strict_load='PASS')
        report['dimensions'] = preflight['dimensions']
        report['forward_calls'] = dict(encoder=0, forecaster=0)
        del payload, state, expected
        shapes = dict(Z=(n, preflight['dimensions']['D']),
                      S=(preflight['dimensions']['K'], preflight['dimensions']['D']),
                      S_tilde=(preflight['dimensions']['K'], preflight['dimensions']['D']),
                      g=(preflight['dimensions']['D'],))
        errors, block_reports = {}, {}
        for split in SEEDS:
            all_moments = {level: {role: Moments() for role in ('current', 'target', 'hbf')} for level in LEVELS}
            h_errors, p_errors, block_results = [], [], []
            for block in plans[split]['blocks']:
                x, y = read_block(block, sources[split], split, w, n,
                                  preflight['splits'][split]['first_input_start_ns'], step)
                arrays = {level: {role: [] for role in ('current', 'target', 'hbf')} for level in LEVELS}
                with torch.no_grad():
                    for start in range(0, ROWS, BATCH):
                        current = model.encode_window(torch.from_numpy(x[start:start+BATCH]).to(device))
                        target = model.encode_window(torch.from_numpy(y[start:start+BATCH]).to(device))
                        forecast = model.forecast(current)
                        report['forward_calls']['encoder'] += 2
                        report['forward_calls']['forecaster'] += 1
                        for level in LEVELS:
                            for role, hierarchy in (('current', current), ('target', target), ('hbf', forecast)):
                                value = hierarchy[level]
                                require(tuple(value.shape) == (BATCH, *shapes[level]) and value.dtype == torch.float32,
                                        f'Unexpected {role}/{level} shape or dtype')
                                array = value.detach().cpu().numpy().copy()
                                require(np.isfinite(array).all(), f'Nonfinite {role}/{level}')
                                arrays[level][role].append(array)
                bh, bp, result = [], [], dict(block=block['block'], levels={})
                for level in LEVELS:
                    values = {role: np.concatenate(parts) for role, parts in arrays[level].items()}
                    h = window_mse(values['hbf'], values['target'])
                    p = window_mse(values['current'], values['target'])
                    bh.append(h)
                    bp.append(p)
                    variation = {}
                    for role, value in values.items():
                        moments = Moments()
                        moments.update(value)
                        variation[role] = moments.summary()
                        all_moments[level][role].update(value)
                    result['levels'][level] = dict(errors=error_summary(h, p), variation=variation)
                h_errors.append(np.stack(bh, axis=1))
                p_errors.append(np.stack(bp, axis=1))
                block_results.append(result)
                print(f'{split}: block {block["block"]+1}/{BLOCKS} complete', flush=True)
            h, p = np.concatenate(h_errors), np.concatenate(p_errors)
            errors[split+'_hbf_mse'], errors[split+'_persistence_mse'] = h, p
            block_reports[split] = block_results
            summaries = {}
            for j, level in enumerate(LEVELS):
                delta = h[:, j].reshape(BLOCKS, ROWS).mean(1)-p[:, j].reshape(BLOCKS, ROWS).mean(1)
                summaries[level] = dict(errors=error_summary(h[:, j], p[:, j]),
                    block_comparison=dict(hbf_lower=int((delta < 0).sum()),
                                          exactly_tied=int((delta == 0).sum()), hbf_higher=int((delta > 0).sum())),
                    variation={role: m.summary() for role, m in all_moments[level].items()})
                summaries[level]['mean_within_block_variation'] = {
                    role: float(np.mean([b['levels'][level]['variation'][role]['centered_population_variance']
                                         for b in block_results]))
                    for role in ('current', 'target', 'hbf')}
            report['splits'][split] = dict(paired_windows=len(h), levels=summaries)
            write_json(report_path, report)
        np.savez_compressed(output/'per_window_errors.npz', levels=np.array(LEVELS), **errors)
        write_json(output/'block_statistics.json', block_reports)
        report['artifacts'] = {name: sha256_file(output/name) for name in (
            'selected_windows.npz', 'selection_plan.json', 'per_window_errors.npz', 'block_statistics.json')}
        # Refuse COMPLETE if any pinned input changed during the diagnostic.
        print('Rechecking pinned inputs...', flush=True)
        for path, expected in tracked.items():
            require(sha256_file(path) == expected, f'Input changed during diagnostic: {path}')
        report['input_sha256'] = tracked
        require(report['forward_calls'] == dict(encoder=256, forecaster=128), 'Incomplete inference budget')
        report['status'] = 'COMPLETE'
        report['scientific_claim_decision'] = 'descriptive_results_only_requires_review'
    except Exception as exc:
        report.update(status='FAILED', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        report['elapsed_seconds'] = time.perf_counter()-started
        write_json(report_path, report)
    print(f'COMPLETE: {report_path}', flush=True)


if __name__ == '__main__':
    main()
