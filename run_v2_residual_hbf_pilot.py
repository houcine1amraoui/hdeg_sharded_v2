#!/usr/bin/env python3
"""Bounded persistence-initialized residual-HBF pilot with a frozen encoder."""
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
from src.scientific_validation.v2.hbf_pilot import update_order, STEPS, EVALUATE_AT, LR
from src.scientific_validation.v2 import residual_hbf_pilot as residual
from src.scientific_validation.v2 import residual_audit
from src.scientific_validation.v2.residual_hbf_pilot import evaluate, train_step
from src.scientific_validation.v2.preflight import require, sha256_file, config_differences, step_ns
from src.scientific_validation.v2.forecast_diagnostic import (
    LEVELS, BLOCKS, ROWS, BATCH, SEEDS, write_json, select_blocks,
    block_shards, read_block,
)

PINNED_CHECKPOINT = '4b036348b014652bb84e0f5b44a2b46374bae00213f4ad07b1d4746ff862157b'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project_root_dir')
    parser.add_argument('--pilot_report', type=Path, required=True, help='Completed original HBF-only pilot report')
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
    output = base/'v2_validation'/'residual_hbf_pilot'/run_id
    output.mkdir(parents=True, exist_ok=False)
    report_path = output/'residual_hbf_report.json'
    started = time.perf_counter()
    report = dict(status='RUNNING', run_id=run_id, configuration=config,
                  protocol='v2_residual_hbf_640_v1', optimizer_steps=0,
                  levels=list(LEVELS), splits={}, source_code_sha256={},
                  budget=dict(total_paired_windows=4096, windows_per_split=2048,
                              blocks_per_split=BLOCKS, windows_per_block=ROWS, batch_size=BATCH),
                  limitations=[
                      'Exploratory persistence-plus-correction parameterization; original backbone with four zeroed final output projections.',
                      'Fresh Adam state; frozen encoder; shuffled sampled training rows differ from the original joint-training regime.',
                      'Training evaluation is in-sample for this pilot; validation has already been inspected and is not a pristine confirmatory set.',
                      'Fixed final step, no validation-based checkpoint selection or stopping.',
                      'Train-mean baseline is fitted only to selected training targets.',
                      'Validation informed this experimental design; subsequent untouched-data confirmation is required.',
                      'Mean-offset removal and centered alignment are descriptive, not deployable corrections or proof of dynamics.',
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
        pilot_hash = pin(args.pilot_report)
        completed = json.loads(args.pilot_report.read_text())
        require(completed['status'] == 'COMPLETE' and completed['protocol'] == 'v2_hbf_only_640_v1',
                'Expected completed original HBF-only pilot')
        require(completed['optimizer_steps'] == STEPS and completed['checkpoint']['sha256'] == PINNED_CHECKPOINT,
                'Pilot checkpoint or final step mismatch')
        for name in ('selected_windows.npz', 'selection_plan.json', 'update_order.npy', 'per_window_errors.npz'):
            pin(args.pilot_report.parent/name, completed['artifacts'][name])
        for old, digest in completed['source_code_sha256'].items():
            source = root/'src'/old.split('/src/', 1)[1] if '/src/' in old else root/Path(old).name
            pin(source, digest)
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
        for source in (__file__, helper.__file__, pilot.__file__, residual.__file__, residual_audit.__file__):
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
        with np.load(args.pilot_report.parent/'selected_windows.npz', allow_pickle=False) as old:
            require(set(old.files) == set(selected_arrays), 'Pilot selection fields changed')
            for key, value in selected_arrays.items():
                require(np.array_equal(old[key], value), f'Pilot selection differs: {key}')
        selection_hash = parent['artifacts']['selected_windows.npz']
        report['selection_reference'] = dict(path=str((args.forecast_report.parent/'selected_windows.npz').resolve()),
                                              sha256=selection_hash)
        print('Selection verified. Loading the original model for the residual-HBF pilot.', flush=True)
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
        model.hbf.requires_grad_(True)
        for obj in (*(type(m) for m in model.modules()), type(model.dbrl.graph_builder)):
            source = inspect.getsourcefile(obj)
            if source and Path(source).resolve().is_relative_to(root):
                report['source_code_sha256'][str(Path(source).resolve())] = pin(source)
        report['checkpoint'].update(epoch=payload['epoch'], strict_load='PASS')
        report['dimensions'] = preflight['dimensions']
        report['optimization'] = dict(steps=STEPS, batch_size=32, passes=10, learning_rate=LR,
            optimizer='Adam', betas=[0.9, 0.999], eps=1e-8, weight_decay=0.0,
            optimizer_state='fresh; original Adam state not resumed', shuffle_seed=42,
            evaluation_steps=list(EVALUATE_AT), checkpoint_rule='final step 640 only')
        report['forward_calls'] = dict(encoder=0, residual_hbf=0)
        report['loss_weights'] = {k: float(v.detach().cpu()) for k,v in model.mo.named_buffers()}
        del payload, state, expected
        shapes = dict(Z=(n, preflight['dimensions']['D']),
                      S=(preflight['dimensions']['K'], preflight['dimensions']['D']),
                      S_tilde=(preflight['dimensions']['K'], preflight['dimensions']['D']),
                      g=(preflight['dimensions']['D'],))
        frozen_before = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()
                         if not name.startswith('hbf.')}
        residual.initialize(model.hbf)
        report['initialization'] = dict(backbone='original checkpoint', zeroed_final_heads=list(residual.HEADS),
                                        prediction_rule='current + hbf(current)', prior_pilot_weights_loaded=False)
        initial_hbf = {name: value.detach().cpu().clone() for name, value in model.hbf.state_dict().items()}
        order_path = args.pilot_report.parent/'update_order.npy'
        order = np.load(order_path, allow_pickle=False)
        require(order.dtype == np.int64 and np.array_equal(order, update_order()), 'Unexpected saved update order')
        report['update_order_reference'] = dict(path=str(order_path.resolve()), sha256=pin(order_path))
        write_json(output/'pilot_plan.json', dict(protocol=report['protocol'], optimization=report['optimization'],
                   initialization=report['initialization'], forecast_reference=report['forecast_reference'],
                   pilot_reference=report['pilot_reference'], selection=report['selection_reference'],
                   update_order=report['update_order_reference'], primary_endpoint=640,
                   uncertainty='descriptive paired block comparisons only; no IID tests or confidence intervals'))
        pin(output/'pilot_plan.json')
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
        means = {k: residual.fit_train_mean(cache['train']['target'][k].numpy()) for k in LEVELS}
        np.savez_compressed(output/'train_mean_vectors.npz', **means)
        report['train_mean_fit'] = dict(split='train', rows=2048, source='selected frozen target representations')
        historical = {}
        with np.load(args.forecast_report.parent/'per_window_errors.npz', allow_pickle=False) as old, \
             np.load(args.pilot_report.parent/'per_window_errors.npz', allow_pickle=False) as previous:
            for split in SEEDS:
                for label, array in (('original_hbf', old[split+'_hbf_mse']),
                                     ('pilot_hbf', previous['step_0640_'+split+'_hbf_mse'])):
                    require(array.shape == (2048, 4) and np.isfinite(array).all() and (array >= 0).all(),
                            'Invalid historical error array')
                    historical[split+'_'+label] = array.copy()
        optimizer = torch.optim.Adam(model.hbf.parameters(), lr=LR, betas=(0.9,0.999), eps=1e-8, weight_decay=0.0)
        curve, updates, saved_errors = {}, [], {}

        def check_frozen():
            for name, value in model.state_dict().items():
                if not name.startswith('hbf.'):
                    require(torch.equal(value.detach().cpu(), frozen_before[name]), f'Frozen state changed: {name}')
            require(all(p.grad is None for name,p in model.named_parameters() if not name.startswith('hbf.')),
                    'Encoder received gradients')

        def measure(at_step):
            check_frozen()
            before = {k: v.detach().cpu().clone() for k,v in model.hbf.state_dict().items()}
            summary, errors = evaluate(model, cache, device, means, step_zero=(at_step == 0))
            report['forward_calls']['residual_hbf'] += 128
            require(all(torch.equal(v.detach().cpu(), before[k]) for k,v in model.hbf.state_dict().items()),
                    'Evaluation mutated HBF state')
            for split in SEEDS:
                for j,k in enumerate(LEVELS):
                    summary[split]['levels'][k]['historical_comparisons'] = {
                        label: residual_audit.compare_to_baseline(errors[split+'_residual_hbf_mse'][:,j],
                                                                 historical[split+'_'+label][:,j])
                        for label in ('original_hbf', 'pilot_hbf')}
            if at_step == 0:
                for split in SEEDS:
                    require(np.array_equal(errors[split+'_residual_hbf_mse'], errors[split+'_persistence_mse']),
                            'Step-zero errors do not exactly equal persistence')
            if at_step == 0:
                consistency = {}
                with np.load(args.forecast_report.parent/'per_window_errors.npz', allow_pickle=False) as old:
                    for split in SEEDS:
                        key = split+'_persistence_mse'
                        value = errors[key]
                        reference = old[key]
                        require(reference.shape == value.shape, 'Baseline error shape changed')
                        good = np.allclose(value, reference, rtol=1e-3, atol=1e-14)
                        consistency[key] = dict(within_tolerance=bool(good), max_absolute_difference=float(np.max(np.abs(value-reference))))
                        require(good, f'Step-zero replay differs from parent: {key}')
                report['step_zero_replay'] = dict(rtol=1e-3, atol=1e-14, arrays=consistency)
            else:
                for split in SEEDS:
                    for pred in ('persistence', 'train_mean'):
                        key = split+'_'+pred+'_mse'
                        require(np.array_equal(errors[key], saved_errors['step_0000_'+key]),
                                'Fixed baseline changed during residual-HBF pilot')
            curve[str(at_step)] = summary
            for key,value in errors.items():
                saved_errors[f'step_{at_step:04d}_'+key] = value
            write_json(output/'evaluation_curve.json', curve)
            np.savez_compressed(output/'per_window_errors.npz', levels=np.array(LEVELS), **saved_errors)
            write_json(output/'training_updates.json', updates)
            report['optimizer_steps'] = at_step
            write_json(report_path, report)
            print(f'Evaluated fixed train/val windows at step {at_step}/{STEPS}', flush=True)

        measure(0)
        for update, indices in enumerate(order, start=1):
            ids = torch.from_numpy(indices)
            current = {k:v.index_select(0, ids).to(device) for k,v in cache['train']['current'].items()}
            target = {k:v.index_select(0, ids).to(device) for k,v in cache['train']['target'].items()}
            with torch.enable_grad():
                loss = train_step(model, current, target, optimizer)
            report['forward_calls']['residual_hbf'] += 1
            report['optimizer_steps'] = update
            updates.append(dict(step=update, pre_update_minibatch_loss=loss['losses'], gradient_norms=loss['gradient_norms']))
            if update % 64 == 0:
                check_frozen()
                print(f'Completed HBF pass {update//64}/10', flush=True)
            if update in EVALUATE_AT:
                measure(update)
        check_frozen()
        require(pilot.cache_digest(cache) == cache_hash, 'Cached representations changed')
        require(any(not torch.equal(value.detach().cpu(), initial_hbf[name]) for name,value in model.hbf.state_dict().items()),
                'HBF did not change during pilot')
        first_gradients = updates[0]['gradient_norms']
        head_names = tuple(name+'.' for name in residual.HEADS)
        require(any(v > 0 for k,v in first_gradients.items() if k.startswith(head_names)),
                'No initial output-projection gradient')
        upstream_nonzero = any(v > 0 for row in updates[1:] for k,v in row['gradient_norms'].items()
                               if not k.startswith(head_names))
        report['gradient_flow'] = dict(initial_output_gradient_nonzero=True,
            initial_upstream_all_zero=all(v == 0 for k,v in first_gradients.items() if not k.startswith(head_names)),
            upstream_nonzero_after_first_update=upstream_nonzero,
            interpretation='Zero upstream gradients on the initial backward pass are expected; later absence is flagged, not evidence of learning.')
        report['checks'] = dict(step_zero_exact_persistence=True, evaluation_state_unchanged=True,
                                fixed_baselines_unchanged=True, inputs_unchanged=True,
                                encoder_and_mo_unchanged=True, encoder_gradients_absent=True,
                                representation_cache_unchanged=True, persistence_unchanged=True, hbf_changed=True)
        require(report['optimizer_steps']==STEPS and report['forward_calls']['encoder']==256 and
                report['forward_calls']['residual_hbf']==1280, 'Incomplete pilot budget')
        for path, digest in tracked.items():
            require(sha256_file(path)==digest, f'Input changed during pilot: {path}')
        torch.save(dict(artifact_type='residual_hbf_pilot_final', prediction_rule='current + hbf(current)', parent_checkpoint_sha256=PINNED_CHECKPOINT,
                        optimizer_steps=STEPS, hbf_state_dict={k:v.detach().cpu() for k,v in model.hbf.state_dict().items()},
                        configuration=config, protocol=report['protocol'], selection_sha256=selection_hash),
                   output/'residual_hbf_final.pt')
        report['input_sha256'] = tracked
        report['artifacts'] = {name:sha256_file(output/name) for name in (
            'pilot_plan.json','evaluation_curve.json','per_window_errors.npz','training_updates.json',
            'train_mean_vectors.npz','residual_hbf_final.pt')}
        report['scientific_claim_decision'] = 'exploratory_residual_parameterization_fixed_endpoint_requires_review'
        report['primary_endpoint'] = dict(step=640, summary=curve['640'])
        report['status'] = 'COMPLETE'
    except Exception as exc:
        report.update(status='FAILED', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        report['elapsed_seconds'] = time.perf_counter()-started
        write_json(report_path, report)
    print(f'COMPLETE: {report_path}', flush=True)


if __name__ == '__main__':
    main()
