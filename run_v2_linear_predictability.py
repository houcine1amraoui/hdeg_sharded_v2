#!/usr/bin/env python3
"""One bounded train-only ridge-linear increment predictability check."""
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
from src.scientific_validation.v2 import linear_predictability as linear
from src.scientific_validation.v2 import residual_audit
from src.scientific_validation.v2.preflight import require, sha256_file, config_differences, step_ns
from src.scientific_validation.v2.forecast_diagnostic import (
    LEVELS, BLOCKS, ROWS, BATCH, SEEDS, write_json, select_blocks,
    block_shards, read_block, window_mse, error_summary, Moments,
)

PINNED_CHECKPOINT = '4b036348b014652bb84e0f5b44a2b46374bae00213f4ad07b1d4746ff862157b'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project_root_dir')
    parser.add_argument('--calibration_report', type=Path, required=True)
    parser.add_argument('--forecast_report', type=Path, help='Optional relocated original forecasting report')
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
    output = base/'v2_validation'/'linear_predictability'/run_id
    output.mkdir(parents=True, exist_ok=False)
    report_path = output/'linear_predictability_report.json'
    started = time.perf_counter()
    report = dict(status='RUNNING', run_id=run_id, configuration=config,
                  protocol='v2_train_linear_increment_16_16_v1', optimizer_steps=0,
                  levels=list(LEVELS), splits={}, source_code_sha256={},
                  budget=dict(training_windows=2048,fit_windows=1024,evaluation_windows=1024,
                              fit_blocks=16,evaluation_blocks=16,rows_per_block=64,
                              encoder_calls=128,hbf_calls=0,ridge_fits=1,optimizer_steps=0),
                  limitations=[
                      'Exploratory frozen-representation diagnostic; encoder originally trained on both groups.',
                      'First 16 selected training blocks fit the probe; last 16 evaluate it. No raw observation overlap.',
                      'Temporal separation does not imply independence; within-block windows overlap.',
                      'No validation/test evaluation, ridge search, or production-model changes.',
                      'Input centering and per-level aggregate scaling use fitting rows only; no target scaling.',
                      'Exactly constant fit levels are excluded even if they vary during evaluation.',
                      'All four current hierarchy levels jointly predict all four increments.',
                      'One fixed regularization strength; failure does not prove absence of nonlinear or other linear predictability.',
                      'Positive probe results do not validate HBF or establish anomaly-detection performance.',
                      'Earlier diagnostics informed the design; this is not pristine confirmatory evidence.',
                      'No p-values, confidence intervals, or cross-level pooled performance claim.'])
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
        calibration_hash=pin(args.calibration_report)
        calibration=json.loads(args.calibration_report.read_text())
        require(calibration['status']=='COMPLETE' and calibration['protocol']=='v2_train_only_lr_grid_5x8_v1',
                'Expected completed train-only calibration')
        require(calibration['checkpoint']['sha256']==PINNED_CHECKPOINT,'Calibration checkpoint mismatch')
        for old,digest in calibration['source_code_sha256'].items():
            source=root/'src'/old.split('/src/',1)[1] if '/src/' in old else root/Path(old).name
            pin(source,digest)
        report['calibration_reference']=dict(path=str(args.calibration_report.resolve()),sha256=calibration_hash,
            run_id=calibration['run_id'])
        args.forecast_report=args.forecast_report or Path(calibration['forecast_reference']['path'])
        parent_hash = pin(args.forecast_report,calibration['forecast_reference']['sha256'])
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
        for source in (__file__, helper.__file__, linear.__file__, residual_audit.__file__):
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
        report['selection_reference']=dict(path=str((args.forecast_report.parent/'selected_windows.npz').resolve()),
            sha256=parent['artifacts']['selected_windows.npz'])
        report['partition']=linear.partition(selected_arrays['train_indices'],selected_arrays['train_block'],w,
            selected_arrays['train_start_ns'],selected_arrays['train_target_end_ns'])
        write_json(output/'linear_plan.json',dict(protocol=report['protocol'],partition=report['partition'],
            selection=report['selection_reference'],ridge=linear.RIDGE,
            preprocessing='fit-only coordinate means; per-level aggregate centered RMS times sqrt(coordinates)',
            predictor='current + ridge(current hierarchy -> increment)',
            baselines=['current','current + fit-only mean increment']))
        pin(output/'linear_plan.json')
        print('Chronological partition verified. Encoding training windows only.',flush=True)
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
        report['forward_calls']=dict(encoder=0,hbf=0)
        del payload,state,expected
        shapes=dict(Z=(n,preflight['dimensions']['D']),S=(preflight['dimensions']['K'],preflight['dimensions']['D']),
            S_tilde=(preflight['dimensions']['K'],preflight['dimensions']['D']),g=(preflight['dimensions']['D'],))
        frozen_before={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
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
        current={k:v.numpy().astype(np.float64) for k,v in cache['train']['current'].items()}
        target={k:v.numpy().astype(np.float64) for k,v in cache['train']['target'].items()}
        fit_current={k:v[:linear.FIT_ROWS] for k,v in current.items()}
        x_fit,stats,details=linear.fit_transform(fit_current)
        increments=np.concatenate([linear.flat(target[k])-linear.flat(current[k]) for k in LEVELS],axis=1)
        report['preprocessing']=details
        print('Solving one float64 ridge system on 1,024 fitting windows...',flush=True)
        fitted,solver=linear.fit_ridge(x_fit,increments[:linear.FIT_ROWS])
        report['solver']=solver
        report['runtime']['ridge_device']='cpu'
        report['runtime']['ridge_dtype']='float64'
        parameters={k+'__input_mean':stats[k]['mean'] for k in LEVELS}
        parameters.update({k+'__divisor':np.array(stats[k]['divisor']) for k in LEVELS})
        parameters.update({k+'__active':np.array(stats[k]['active']) for k in LEVELS})
        parameters['mean_increment']=fitted['mean_increment']
        np.savez_compressed(output/'fit_statistics.npz',**parameters)
        errors={}
        for name,sl in [('fit',slice(0,linear.FIT_ROWS)),('evaluation',slice(linear.FIT_ROWS,None))]:
            c={k:v[sl] for k,v in current.items()};t={k:v[sl] for k,v in target.items()}
            x=x_fit if name=='fit' else linear.transform(c,stats)
            prediction=linear.predict(fitted,x)
            summary,per=linear.assess(c,t,prediction,fitted['mean_increment'],selected_arrays['train_block'][sl])
            report['splits'][name]=dict(windows=len(x),levels=summary,
                transformed_input_mean_square=float(np.mean(x*x)),
                transformed_input_max_abs=float(np.max(np.abs(x))))
            errors.update({name+'__'+key:value for key,value in per.items()})
            print(f'Assessed {name}: {len(x)} paired windows.',flush=True)
        np.savez_compressed(output/'per_window_errors.npz',**errors)
        write_json(output/'block_comparisons.json',{group:{k:v['block_comparisons'] for k,v in data['levels'].items()}
            for group,data in report['splits'].items()})
        for name,value in model.state_dict().items():
            require(torch.equal(value.detach().cpu(),frozen_before[name]),f'Model state changed: {name}')
        require(all(p.grad is None for p in model.parameters()),'Model gradients present')
        require(report['forward_calls']==dict(encoder=128,hbf=0),'Unexpected inference budget')
        for path,digest in tracked.items():require(sha256_file(path)==digest,f'Input changed: {path}')
        report['checks']=dict(full_hdeg_state_unchanged=True,model_gradients_absent=True,
            fit_evaluation_raw_overlap=False,fit_only_preprocessing=True,fit_only_mean_increment=True,
            input_files_unchanged=True,validation_evaluated=False,test_evaluated=False,checkpoint_saved=False)
        report['input_sha256']=tracked
        report['artifacts']={name:sha256_file(output/name) for name in (
            'linear_plan.json','fit_statistics.npz','per_window_errors.npz','block_comparisons.json')}
        report['scientific_claim_decision']='bounded_frozen_encoder_linear_probe_requires_review'
        report['status']='COMPLETE'
    except Exception as exc:
        report.update(status='FAILED',error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        report['elapsed_seconds']=time.perf_counter()-started
        write_json(report_path,report)
    print(f'COMPLETE: {report_path}',flush=True)


if __name__=='__main__':main()
