#!/usr/bin/env python3
"""Eight-update, train-only instrumented replay of residual-HBF initialization."""
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
from src.scientific_validation.v2.hbf_pilot import update_order, LR
from src.scientific_validation.v2 import optimization_replay as replay
from src.scientific_validation.v2.optimization_replay import STEPS, PROBE_ROWS
from src.scientific_validation.v2 import residual_hbf_pilot as residual
from src.scientific_validation.v2 import residual_audit
from src.scientific_validation.v2.preflight import require, sha256_file, config_differences, step_ns
from src.scientific_validation.v2.forecast_diagnostic import (
    LEVELS, BLOCKS, ROWS, BATCH, SEEDS, write_json, select_blocks,
    block_shards, read_block,
)

PINNED_CHECKPOINT = '4b036348b014652bb84e0f5b44a2b46374bae00213f4ad07b1d4746ff862157b'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project_root_dir')
    parser.add_argument('--residual_report', type=Path, required=True, help='Completed residual-HBF pilot report')
    parser.add_argument('--pilot_report', type=Path, help='Optional relocated original HBF-only pilot report')
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
    output = base/'v2_validation'/'optimization_replay'/run_id
    output.mkdir(parents=True, exist_ok=False)
    report_path = output/'optimization_replay_report.json'
    started = time.perf_counter()
    report = dict(status='RUNNING', run_id=run_id, configuration=config,
                  protocol='v2_train_only_8_update_replay_v1', optimizer_steps=0,
                  levels=list(LEVELS), splits={}, source_code_sha256={},
                  budget=dict(training_pool=2048, batch_size=32, optimizer_updates=8,
                              gradient_rows=256, fixed_probe_rows=64, encoder_calls=128,
                              residual_hbf_calls=34, validation_windows=0),
                  limitations=[
                      'Instrumented replay on a disposable model; no checkpoint is saved or promoted.',
                      'Only training data are evaluated; prior validation findings motivated this measurement.',
                      'First eight saved minibatches only; no hyperparameter search or early selection.',
                      'The fixed probe is the first selected training block; it is not independent or held out.',
                      'Probe and update rows may overlap; exact overlap is recorded.',
                      'Train-only observations do not establish validation or anomaly-detection performance.',
                      'Head contribution decomposition uses float64; fused float32 rounding is reported separately.',
                      'Weight and bias contributions can cancel; their separate energies must not be summed without the cross term.',
                      'Recorded gradients/updates diagnose scales, not a unique cause or a recommended new learning rate.',
                      'No IID tests, confidence intervals, or automatic model changes.'])
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
        replay_parent_hash = pin(args.residual_report)
        replay_parent = json.loads(args.residual_report.read_text())
        require(replay_parent['status']=='COMPLETE' and replay_parent['protocol']=='v2_residual_hbf_640_v1'
                and replay_parent['optimizer_steps']==640, 'Expected completed residual-HBF pilot')
        require(replay_parent['checkpoint']['sha256']==PINNED_CHECKPOINT, 'Residual pilot checkpoint mismatch')
        pin(args.residual_report.parent/'training_updates.json', replay_parent['artifacts']['training_updates.json'])
        original_updates = json.loads((args.residual_report.parent/'training_updates.json').read_text())
        require([x['step'] for x in original_updates]==list(range(1,641)), 'Incomplete original update log')
        for old, digest in replay_parent['source_code_sha256'].items():
            source = root/'src'/old.split('/src/',1)[1] if '/src/' in old else root/Path(old).name
            pin(source,digest)
        report['residual_reference'] = dict(path=str(args.residual_report.resolve()), sha256=replay_parent_hash,
                                            run_id=replay_parent['run_id'])
        args.pilot_report = args.pilot_report or Path(replay_parent['pilot_reference']['path'])
        pilot_hash = pin(args.pilot_report, replay_parent['pilot_reference']['sha256'])
        completed = json.loads(args.pilot_report.read_text())
        require(completed['status'] == 'COMPLETE' and completed['protocol'] == 'v2_hbf_only_640_v1',
                'Expected completed original HBF-only pilot')
        require(completed['optimizer_steps'] == 640 and completed['checkpoint']['sha256'] == PINNED_CHECKPOINT,
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
        for source in (__file__, helper.__file__, pilot.__file__, residual.__file__, residual_audit.__file__, replay.__file__):
            report['source_code_sha256'][str(Path(source).resolve())] = pin(source)
        pin(Path('configs/config.yaml'))
        semantics_path = Path(config['project_root_dir'])/'configs'/'hdeg'/'behavioral_states.yaml'
        pin(semantics_path, preflight['semantics']['behavioral_states_sha256'])
        pin(base/'devices.json', preflight['semantics']['devices_sha256'])
        w, n = preflight['dimensions']['W'], preflight['dimensions']['N']
        step = step_ns(config['preprocessing']['downsample_freq'])
        from run_dbrl_sharded import load_manifest, resolve_shard_path
        plans, sources, selected_arrays = {}, {}, {}
        for split in ('train',):
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
            require({k for k in original.files if k.startswith('train_')} == set(selected_arrays), 'Selection fields mismatch')
            for key, value in selected_arrays.items():
                require(np.array_equal(original[key], value), f'Frozen selection differs: {key}')
        original_plan = json.loads((args.forecast_report.parent/'selection_plan.json').read_text())
        for split in ('train',):
            original_hashes = {s['shard_index']: s['sha256'] for s in original_plan['splits'][split]['shards']}
            for shard in sources[split]:
                require(shard['sha256'] == original_hashes[shard['shard_index']], 'Source changed since forecasting diagnostic')
        with np.load(args.pilot_report.parent/'selected_windows.npz', allow_pickle=False) as old:
            require({k for k in old.files if k.startswith('train_')} == set(selected_arrays), 'Pilot selection fields changed')
            for key, value in selected_arrays.items():
                require(np.array_equal(old[key], value), f'Pilot selection differs: {key}')
        selection_hash = parent['artifacts']['selected_windows.npz']
        report['selection_reference'] = dict(path=str((args.forecast_report.parent/'selected_windows.npz').resolve()),
                                              sha256=selection_hash)
        print('Selection verified. Loading the original model for the train-only instrumented replay.', flush=True)
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
        report['optimization'] = dict(steps=8, batch_size=32, learning_rate=LR,
            optimizer='Adam', betas=[0.9,0.999], eps=1e-8, weight_decay=0.0,
            optimizer_state='fresh', order='first eight minibatches of the original saved order',
            checkpoint_rule='none saved', evaluations='probe at 0..8; each update minibatch before/after its update')
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
        order_path = args.pilot_report.parent/'update_order.npy'
        order = np.load(order_path, allow_pickle=False)
        require(order.dtype == np.int64 and np.array_equal(order, update_order()), 'Unexpected saved update order')
        report['update_order_reference'] = dict(path=str(order_path.resolve()), sha256=pin(order_path))
        write_json(output/'pilot_plan.json', dict(protocol=report['protocol'], optimization=report['optimization'],
                   initialization=report['initialization'], forecast_reference=report['forecast_reference'],
                   pilot_reference=report['pilot_reference'], selection=report['selection_reference'],
                   update_order=report['update_order_reference'], residual_reference=report['residual_reference'],
                   probe_pool_rows=list(range(PROBE_ROWS)), update_pool_rows=order[:STEPS].tolist(),
                   probe_update_overlap_rows=np.intersect1d(np.arange(PROBE_ROWS),order[:STEPS].ravel()).tolist(),
                   uncertainty='descriptive paired block comparisons only; no IID tests or confidence intervals'))
        pin(output/'pilot_plan.json')
        cache = {}
        for split in ('train',):
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
        cache_hash = replay.cache_digest(cache)
        report['representation_cache'] = dict(storage='RAM only; train only', sha256=cache_hash,
            bytes=sum(v.numel()*v.element_size() for values in cache['train'].values() for v in values.values()))
        optimizer = torch.optim.Adam(model.hbf.parameters(), lr=LR, betas=(0.9,0.999), eps=1e-8, weight_decay=0.0)
        probe_current = {k:v[:PROBE_ROWS].to(device) for k,v in cache['train']['current'].items()}
        probe_target = {k:v[:PROBE_ROWS].to(device) for k,v in cache['train']['target'].items()}
        updates, curve, errors = [], {}, {}

        def check_frozen():
            for name, value in model.state_dict().items():
                if not name.startswith('hbf.'):
                    require(torch.equal(value.detach().cpu(), frozen_before[name]), f'Frozen state changed: {name}')
            require(all(p.grad is None for name,p in model.named_parameters() if not name.startswith('hbf.')),
                    'Encoder received gradients')

        def measure(at_step):
            before = {k:v.detach().cpu().clone() for k,v in model.hbf.state_dict().items()}
            summary, a = replay.probe(model,probe_current,probe_target,step_zero=(at_step==0))
            report['forward_calls']['residual_hbf'] += 2
            require(all(torch.equal(v.detach().cpu(),before[k]) for k,v in model.hbf.state_dict().items()),
                    'Probe mutated HBF state')
            if at_step == 0:
                require(np.array_equal(a[:,:,0],a[:,:,1]), 'Step-zero probe differs from persistence')
            else:
                require(np.array_equal(a[:,:,1],errors['probe_0000'][:,:,1]), 'Probe persistence changed')
            curve[str(at_step)] = summary
            errors[f'probe_{at_step:04d}'] = a
            check_frozen()
            write_json(output/'probe_curve.json',curve)
            write_json(output/'instrumented_updates.json',updates)
            np.savez_compressed(output/'per_window_errors.npz', levels=np.array(LEVELS),
                                predictors=np.array(['residual_hbf','persistence']), **errors)
            write_json(report_path,report)

        measure(0)
        for update, indices in enumerate(order[:STEPS],start=1):
            ids = torch.from_numpy(indices)
            current = {k:v.index_select(0,ids).to(device) for k,v in cache['train']['current'].items()}
            target = {k:v.index_select(0,ids).to(device) for k,v in cache['train']['target'].items()}
            record, before, after = replay.update(model,current,target,optimizer)
            report['forward_calls']['residual_hbf'] += 2
            report['optimizer_steps'] = update
            record.update(step=update, pool_rows=indices.tolist(),
                          global_window_indices=selected_arrays['train_indices'][indices].tolist())
            reference = original_updates[update-1]['pre_update_minibatch_loss']
            record['historical_loss_replay'] = dict(rtol=1e-3,atol=1e-14,
                comparisons={key:dict(original=reference[key],replayed=value,
                    within_tolerance=bool(np.isclose(value,reference[key],rtol=1e-3,atol=1e-14)))
                    for key,value in record['losses'].items()})
            updates.append(record)
            errors[f'batch_{update:04d}_before'] = before
            errors[f'batch_{update:04d}_after'] = after
            measure(update)
            print(f'Instrumented update {update}/{STEPS}; same-batch and fixed-probe measurements saved.',flush=True)
        check_frozen()
        require(replay.cache_digest(cache)==cache_hash,'Cached representations changed')
        require(report['forward_calls']==dict(encoder=128,residual_hbf=34) and report['optimizer_steps']==8,
                'Incomplete replay budget')
        for path,digest in tracked.items():
            require(sha256_file(path)==digest,f'Input changed during replay: {path}')
        faithful = all(x['within_tolerance'] for row in updates
                       for x in row['historical_loss_replay']['comparisons'].values())
        report['historical_replay_matches'] = faithful
        report['checks'] = dict(encoder_and_mo_unchanged=True, encoder_gradients_absent=True,
            representation_cache_unchanged=True, step_zero_exact_persistence=True,
            probe_state_unchanged=True, probe_persistence_unchanged=True,
            input_files_unchanged=True, validation_evaluated=False, trained_checkpoint_saved=False)
        report['input_sha256'] = tracked
        report['artifacts'] = {name:sha256_file(output/name) for name in (
            'pilot_plan.json','probe_curve.json','instrumented_updates.json','per_window_errors.npz')}
        report['scientific_claim_decision'] = ('train_only_optimization_measurement_requires_review' if faithful else
            'historical_replay_mismatch_do_not_attribute_measurements_to_original_trajectory')
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
