#!/usr/bin/env python3
"""Bounded train-only learning-rate calibration from identical residual-HBF initialization."""
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
from src.scientific_validation.v2 import lr_calibration as calibration
from src.scientific_validation.v2.lr_calibration import STEPS, RATES, PROBE_AT
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
    parser.add_argument('--replay_report', type=Path, required=True, help='Completed eight-update instrumented replay report')
    parser.add_argument('--residual_report', type=Path, help='Optional relocated residual-HBF pilot report')
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
    output = base/'v2_validation'/'lr_calibration'/run_id
    output.mkdir(parents=True, exist_ok=False)
    report_path = output/'lr_calibration_report.json'
    started = time.perf_counter()
    report = dict(status='RUNNING', run_id=run_id, configuration=config,
                  protocol='v2_train_only_lr_grid_5x8_v1', optimizer_steps=0,
                  levels=list(LEVELS), splits={}, source_code_sha256={},
                  budget=dict(training_pool=2048,batch_size=32,candidates=5,updates_per_candidate=8,
                              total_optimizer_updates=40,update_rows_per_candidate=256,probe_rows=256,
                              encoder_calls=128,residual_hbf_calls=200,validation_windows=0),
                  limitations=[
                      'Train-only calibration, not forecasting validation; earlier validation motivated this design.',
                      'Five fixed rates; eight updates each; identical initialization and fresh Adam for every candidate.',
                      'Probe rows excluded from these updates may overlap their raw temporal observations.',
                      'All rows belong to the original encoder training split; probe is not an independent holdout.',
                      'The probe differs from the earlier near-constant first-block probe by an outcome-independent saved-order rule.',
                      'No automatic winner, configuration change, trained checkpoint, or validation evaluation.',
                      'No extension of the grid or budget based on outcomes; all candidates reported.',
                      'Raw denominators retained; tiny baseline errors can produce unstable ratios.',
                      'A smaller correction or lower same-batch error does not by itself establish transferable dynamics.',
                      'No p-values, confidence intervals, or cross-level pooling.'])
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
        calibration_parent_hash=pin(args.replay_report)
        calibration_parent=json.loads(args.replay_report.read_text())
        require(calibration_parent['status']=='COMPLETE' and calibration_parent['protocol']=='v2_train_only_8_update_replay_v1'
                and calibration_parent['optimizer_steps']==8,'Expected completed instrumented replay')
        require(calibration_parent['checkpoint']['sha256']==PINNED_CHECKPOINT,'Replay checkpoint mismatch')
        for old,digest in calibration_parent['source_code_sha256'].items():
            source=root/'src'/old.split('/src/',1)[1] if '/src/' in old else root/Path(old).name
            pin(source,digest)
        report['replay_reference']=dict(path=str(args.replay_report.resolve()),sha256=calibration_parent_hash,
            run_id=calibration_parent['run_id'],historical_replay_matches=calibration_parent['historical_replay_matches'])
        args.residual_report=args.residual_report or Path(calibration_parent['residual_reference']['path'])
        replay_parent_hash = pin(args.residual_report,calibration_parent['residual_reference']['sha256'])
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
        for source in (__file__, helper.__file__, pilot.__file__, residual.__file__, residual_audit.__file__, replay.__file__, calibration.__file__):
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
        print('Selection verified. Loading the original model for train-only learning-rate calibration.', flush=True)
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
        report['optimization']=dict(rates=list(RATES),steps_per_candidate=8,batch_size=32,
            optimizer='Adam',betas=[0.9,0.999],eps=1e-8,weight_decay=0.0,
            optimizer_state='fresh for every candidate',initialization='identical original backbone and zero heads',
            probe_steps=list(PROBE_AT),selection_rule='none; report every candidate; no configuration update')
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
        update_rows,probe_rows,overlap=calibration.selections(order,selected_arrays['train_indices'],w)
        report['probe_selection']=dict(rule='saved order batches 9 through 16, flattened',raw_overlap=overlap,
            row_overlap=0,block_counts=np.bincount(selected_arrays['train_block'][probe_rows],minlength=32).tolist())
        initial_hbf={k:v.detach().cpu().clone() for k,v in model.hbf.state_dict().items()}
        write_json(output/'calibration_plan.json',dict(protocol=report['protocol'],optimization=report['optimization'],
            initialization=report['initialization'],replay_reference=report['replay_reference'],
            selection=report['selection_reference'],update_order=report['update_order_reference'],
            update_pool_rows=update_rows.tolist(),probe_pool_rows=probe_rows.tolist(),
            update_global_indices=selected_arrays['train_indices'][update_rows].tolist(),
            probe_global_indices=selected_arrays['train_indices'][probe_rows].tolist(),
            probe_selection=report['probe_selection']))
        pin(output/'calibration_plan.json')
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
        probe_ids=torch.from_numpy(probe_rows)
        probe_current={k:v.index_select(0,probe_ids).to(device) for k,v in cache['train']['current'].items()}
        probe_target={k:v.index_select(0,probe_ids).to(device) for k,v in cache['train']['target'].items()}
        updates,curve,errors={}, {}, {}
        fixed_baseline=None
        first_candidate_batch_baselines={}
        report['candidate_steps']={}

        def check_frozen():
            for name,value in model.state_dict().items():
                if not name.startswith('hbf.'):
                    require(torch.equal(value.detach().cpu(),frozen_before[name]),f'Frozen state changed: {name}')
            require(all(p.grad is None for name,p in model.named_parameters() if not name.startswith('hbf.')),
                    'Encoder received gradients')

        def save():
            write_json(output/'probe_curve.json',curve)
            write_json(output/'instrumented_updates.json',updates)
            np.savez_compressed(output/'per_window_errors.npz',levels=np.array(LEVELS),
                predictors=np.array(['residual_hbf','persistence']),**errors)
            write_json(report_path,report)

        def measure(label,at_step):
            nonlocal fixed_baseline
            before={k:v.detach().cpu().clone() for k,v in model.hbf.state_dict().items()}
            summary,a=calibration.probe(model,probe_current,probe_target,step_zero=(at_step==0))
            report['forward_calls']['residual_hbf']+=8
            require(all(torch.equal(v.detach().cpu(),before[k]) for k,v in model.hbf.state_dict().items()),
                    'Probe mutated HBF state')
            if fixed_baseline is None:fixed_baseline=a[:,:,1].copy()
            require(np.array_equal(a[:,:,1],fixed_baseline),'Probe baseline changed across candidates')
            if at_step==0:require(np.array_equal(a[:,:,0],fixed_baseline),'Step zero is not persistence')
            curve[label][str(at_step)]=summary
            errors[f'{label}_probe_{at_step:04d}']=a
            check_frozen()
            save()

        for rate in RATES:
            label=f'lr_{rate:.0e}'
            optimizer=calibration.reset_candidate(model,initial_hbf,rate)
            curve[label]={};updates[label]=[];report['candidate_steps'][label]=0
            measure(label,0)
            for update,indices in enumerate(update_rows,start=1):
                ids=torch.from_numpy(indices)
                current={k:v.index_select(0,ids).to(device) for k,v in cache['train']['current'].items()}
                target={k:v.index_select(0,ids).to(device) for k,v in cache['train']['target'].items()}
                record,before,after=replay.update(model,current,target,optimizer)
                report['forward_calls']['residual_hbf']+=2
                report['optimizer_steps']+=1
                report['candidate_steps'][label]=update
                require(np.array_equal(before[:,:,1],after[:,:,1]),'Minibatch baseline changed')
                if rate==RATES[0]:first_candidate_batch_baselines[update]=before[:,:,1].copy()
                require(np.array_equal(before[:,:,1],first_candidate_batch_baselines[update]),
                        'Minibatch baseline changed across candidates')
                record.update(step=update,learning_rate=rate,pool_rows=indices.tolist())
                if rate==RATES[0]:
                    reference=original_updates[update-1]['pre_update_minibatch_loss']
                    record['historical_control_replay']={key:dict(original=reference[key],replayed=value,
                        within_tolerance=bool(np.isclose(value,reference[key],rtol=1e-3,atol=1e-14)))
                        for key,value in record['losses'].items()}
                updates[label].append(record)
                errors[f'{label}_batch_{update:04d}_before']=before
                errors[f'{label}_batch_{update:04d}_after']=after
                check_frozen()
                if update in PROBE_AT:measure(label,update)
            require(all(int(state['step'].item())==STEPS for state in optimizer.state.values()),
                    'Candidate optimizer did not execute eight steps')
            print(f'Completed {label}: 8 updates; fixed probe at 0,1,8.',flush=True)
            del optimizer
        require(replay.cache_digest(cache)==cache_hash,'Cached representations changed')
        require(report['forward_calls']==dict(encoder=128,residual_hbf=200) and report['optimizer_steps']==40,
                'Incomplete calibration budget')
        for path,digest in tracked.items():require(sha256_file(path)==digest,f'Input changed: {path}')
        comparison=calibration.comparison_table(curve,updates)
        write_json(output/'candidate_comparison.json',comparison)
        report['candidate_comparison']=comparison
        report['historical_control_replay_matches']=all(v['within_tolerance'] for row in updates['lr_1e-04']
            for v in row['historical_control_replay'].values())
        report['historical_control_replay_tolerance']=dict(rtol=1e-3,atol=1e-14,
            interpretation='Diagnostic only; mismatches retained. Candidate comparisons use the shared cache within this run.')
        report['checks']=dict(identical_initialization=True,fresh_optimizer_per_candidate=True,
            encoder_and_mo_unchanged=True,encoder_gradients_absent=True,representation_cache_unchanged=True,
            fixed_baselines_unchanged=True,step_zero_exact_persistence=True,probe_state_unchanged=True,
            row_disjoint_probe=True,input_files_unchanged=True,validation_evaluated=False,trained_checkpoint_saved=False)
        report['input_sha256']=tracked
        report['artifacts']={name:sha256_file(output/name) for name in (
            'calibration_plan.json','probe_curve.json','instrumented_updates.json','per_window_errors.npz','candidate_comparison.json')}
        report['scientific_claim_decision']='train_only_calibration_requires_review_no_automatic_selection'
        report['status']='COMPLETE'
    except Exception as exc:
        report.update(status='FAILED',error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        report['elapsed_seconds']=time.perf_counter()-started
        write_json(report_path,report)
    print(f'COMPLETE: {report_path}',flush=True)


if __name__=='__main__':main()
