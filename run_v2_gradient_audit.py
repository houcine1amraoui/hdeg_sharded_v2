#!/usr/bin/env python3
"""Standalone Level 2 gradient audit: 128 training pairs; zero optimizer steps."""
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
from src.scientific_validation.v2 import gradient_audit as helper
from src.scientific_validation.v2 import forecast_diagnostic as forecast_helper
from src.scientific_validation.v2.preflight import require, sha256_file, config_differences, step_ns
from src.scientific_validation.v2.forecast_diagnostic import write_json, read_block, block_shards, select_blocks
from src.scientific_validation.v2.gradient_audit import (
    BLOCK_IDS, BATCH_SIZE, audit_batch, interpretation_flags,
)

CHECKPOINT_SHA256 = '4b036348b014652bb84e0f5b44a2b46374bae00213f4ad07b1d4746ff862157b'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project_root_dir')
    parser.add_argument('--forecast_report', type=Path, required=True,
                        help='Completed diagnostic_report.json; its companion files must be alongside it')
    parser.add_argument('--preflight_report', type=Path, help='Optional relocated copy of the linked preflight')
    parser.add_argument('--checkpoint', type=Path, help='Optional relocated copy of the SAME checkpoint')
    parser.add_argument('--device', choices=('cpu', 'cuda'))
    args = parser.parse_args()
    with open('configs/config.yaml', encoding='utf-8') as stream:
        config = yaml.safe_load(stream)
    require(isinstance(config, dict), 'Configuration must be a mapping')
    if args.project_root_dir is not None:
        config['project_root_dir'] = args.project_root_dir
    base = Path(get_processed_folder(config)).resolve()
    root = Path(__file__).resolve().parent
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8]
    output = base/'v2_validation'/'gradient_audit'/run_id
    output.mkdir(parents=True, exist_ok=False)
    report_path = output/'gradient_report.json'
    started = time.perf_counter()
    report = dict(status='RUNNING', run_id=run_id, configuration=config,
                  protocol='v2_gradient_4x32_v1', optimizer_steps=0,
                  budget=dict(split='train', block_ids=list(BLOCK_IDS), offset=0,
                              batch_size=BATCH_SIZE, minibatches=4, paired_windows=128,
                              encoder_calls=8, forecaster_calls=4, autograd_queries=60),
                  source_code_sha256={}, completed_batches=0,
                  limitations=[
                      'Exploratory protocol chosen after the forecasting diagnostic; blocks chosen by position, not outcome.',
                      'Gradients describe this frozen checkpoint and four minibatches, not historical optimization.',
                      'No optimizer steps, Adam-state interpretation, convergence claim or collapse diagnosis.',
                      'Branch isolation uses detach only for derivative measurements; the full objective is unchanged.',
                      'Absent branch-specific gradients can be structurally expected; zero gradients can be data-dependent.',
                      'No near-zero threshold or module-ranking claim is imposed; norms depend on scale and parameter count.',
                      'Top-k neighbor selection is discrete; gradients describe the selected topology, not topology changes.',
                      'Ancestor preflight does not pin every transitive source; additional imported project sources are pinned now.'])
    write_json(report_path, report)
    print(f'Gradient audit report: {report_path}', flush=True)
    tracked = {}

    def pin(path, expected=None):
        path = Path(path).resolve()
        digest = sha256_file(path)
        require(expected is None or expected == digest, f'Hash mismatch: {path}')
        require(str(path) not in tracked or tracked[str(path)] == digest, f'Input changed: {path}')
        tracked[str(path)] = digest
        return digest

    batches = []
    try:
        parent_hash = pin(args.forecast_report)
        parent = json.loads(args.forecast_report.read_text())
        require(parent['status'] == 'COMPLETE' and parent['protocol'] == 'v2_forecast_32x64_v1',
                'Expected completed frozen forecasting diagnostic')
        require(parent['checkpoint']['sha256'] == CHECKPOINT_SHA256, 'Unexpected checkpoint lineage')
        report['forecast_reference'] = dict(path=str(args.forecast_report.resolve()),
                                           sha256=parent_hash, run_id=parent['run_id'])
        for name in ('selection_plan.json', 'selected_windows.npz', 'per_window_errors.npz', 'block_statistics.json'):
            pin(args.forecast_report.parent/name, parent['artifacts'][name])
        plan = json.loads((args.forecast_report.parent/'selection_plan.json').read_text())
        preflight_path = args.preflight_report or Path(parent['preflight']['path'])
        preflight_hash = pin(preflight_path, parent['preflight']['sha256'])
        preflight = json.loads(preflight_path.read_text())
        require(preflight['status'] == 'COMPLETE', 'Preflight not COMPLETE')
        require(preflight['checkpoint']['sha256'] == CHECKPOINT_SHA256 and
                plan['checkpoint_sha256'] == CHECKPOINT_SHA256 and
                plan['preflight_sha256'] == preflight_hash, 'Lineage mismatch')
        report['preflight_reference'] = dict(path=str(preflight_path.resolve()), sha256=preflight_hash)
        for reference in (parent, preflight):
            require(all(d['category'] == 'path_only' for d in config_differences(config, reference['configuration'])),
                    'Non-path configuration changes since diagnostic/preflight')
        for old, expected in parent['source_code_sha256'].items():
            source = root/'src'/old.split('/src/', 1)[1] if '/src/' in old else root/Path(old).name
            report['source_code_sha256'][str(source.resolve())] = pin(source, expected)
        for source in (__file__, helper.__file__):
            report['source_code_sha256'][str(Path(source).resolve())] = pin(source)
        # Ensure the shared reader actually imported is the pinned diagnostic helper.
        require(str(Path(forecast_helper.__file__).resolve()) in tracked, 'Unexpected forecast helper import')
        pin('configs/config.yaml')
        checkpoint = args.checkpoint or Path(config['hdeg']['hbf']['checkpoint'])
        if not checkpoint.is_absolute():
            checkpoint = base/'hdeg_checkpoints'/checkpoint
        report['checkpoint'] = dict(path=str(checkpoint.resolve()), sha256=pin(checkpoint, CHECKPOINT_SHA256))
        semantics_path = Path(config['project_root_dir'])/'configs'/'hdeg'/'behavioral_states.yaml'
        pin(semantics_path, preflight['semantics']['behavioral_states_sha256'])
        pin(base/'devices.json', preflight['semantics']['devices_sha256'])
        train = preflight['splits']['train']
        pin(base/'windows'/'train'/'manifest.json', train['manifest_sha256'])
        w, n = preflight['dimensions']['W'], preflight['dimensions']['N']
        step = step_ns(config['preprocessing']['downsample_freq'])
        all_blocks = plan['splits']['train']['blocks']
        require(all_blocks == select_blocks(train['sample_count'], w, 42), 'Parent block selection mismatch')
        blocks = [dict(all_blocks[i], end_index=all_blocks[i]['start_index']+BATCH_SIZE,
                       observation_end_exclusive=all_blocks[i]['start_index']+BATCH_SIZE+w) for i in BLOCK_IDS]
        shards = [dict(s, file=str(base/'windows'/'train'/Path(s['file']).name))
                  for s in plan['splits']['train']['shards']]
        needed = {s['shard_index'] for b in blocks for s in block_shards(b, shards)}
        shards = [s for s in shards if s['shard_index'] in needed]
        for shard in shards:
            pin(shard['file'], shard['sha256'])
        with np.load(args.forecast_report.parent/'selected_windows.npz', allow_pickle=False) as saved:
            for block in blocks:
                idx = np.arange(block['start_index'], block['end_index'], dtype=np.int64)
                offset = block['block']*64
                require(np.array_equal(saved['train_indices'][offset:offset+BATCH_SIZE], idx), 'Parent indices mismatch')
        write_json(output/'gradient_plan.json', dict(protocol=report['protocol'], blocks=blocks,
                   shards=shards, forecast_report_sha256=parent_hash, checkpoint_sha256=CHECKPOINT_SHA256,
                   rule='First 32 rows of training blocks 0,10,21,31; no outcome-based filtering'))
        pin(output/'gradient_plan.json')
        import torch
        from src.training.train_hdeg_e2e import build_live_model
        from src.common.graph.semantics import load_behavioral_state_config
        device = torch.device(args.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        torch.manual_seed(42)
        np.random.seed(42)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
        state = payload['model_state_dict']
        require(not any(d['category'] == 'experiment_critical' for d in config_differences(config, payload['config'])),
                'Checkpoint/configuration mismatch')
        semantics = load_behavioral_state_config(config_path=semantics_path, devices_path=base/'devices.json')
        model = build_live_model(config=config, behavioral_config=semantics,
                    compatibility_mask=semantics.torch_mask(dtype=torch.float32, device=torch.device('cpu'), clone=True),
                    device=torch.device('cpu'))
        for name, value in model.named_buffers():
            require(name in state and value.dtype == state[name].dtype and torch.equal(value, state[name]),
                    f'Fixed buffer mismatch: {name}')
        expected = model.state_dict()
        require(set(expected) == set(state), 'Model state keys mismatch')
        for name, value in expected.items():
            require(value.shape == state[name].shape and value.dtype == state[name].dtype and
                    torch.isfinite(state[name]).all().item(), f'Invalid checkpoint tensor: {name}')
        model.load_state_dict(state, strict=True)
        model.to(device).train()
        for obj in (build_live_model, load_behavioral_state_config, get_processed_folder,
                    *(type(m) for m in model.children())):
            source = str(Path(inspect.getsourcefile(obj)).resolve())
            require(source in tracked, f'Imported model source was not pinned by the parent: {source}')
            pin(source, tracked[source])
        # Training mode supports native CuDNN GRU backward. Reject stochastic
        # dropout/running-stat layers instead of silently changing the model.
        for module in model.modules():
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                require(not module.track_running_stats, 'Running-stat BatchNorm is outside this audit protocol')
            if isinstance(module, torch.nn.modules.dropout._DropoutNd):
                require(module.p == 0, 'Nonzero dropout is outside this audit protocol')
            dropout = getattr(module, 'dropout', 0)
            if isinstance(dropout, (int, float)):
                require(dropout == 0, 'Nonzero functional/recurrent dropout is outside this audit protocol')
        for obj in (build_live_model, load_behavioral_state_config, get_processed_folder,
                    *(type(m) for m in model.modules())):
            source = inspect.getsourcefile(obj)
            if source and Path(source).resolve().is_relative_to(root):
                report['source_code_sha256'][str(Path(source).resolve())] = pin(source)
        # GraphBuilder is not an nn.Module; explicitly include its imported source.
        source = inspect.getsourcefile(type(model.dbrl.graph_builder))
        report['source_code_sha256'][str(Path(source).resolve())] = pin(source)
        before = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        report['checkpoint'].update(epoch=payload['epoch'], strict_load='PASS')
        del payload, state, expected
        report['runtime'] = dict(python=platform.python_version(), numpy=np.__version__, torch=torch.__version__,
                                device=str(device), model_mode='train; dropout zero; no optimizer',
                                gradient_dtype='float32', summary_dtype='float64', tf32=False,
                                bitwise_cross_platform_reproducibility='not_guaranteed')
        report['parameter_inventory'] = {name: dict(shape=list(p.shape), elements=p.numel(), requires_grad=p.requires_grad)
                                         for name, p in model.named_parameters()}
        for block in blocks:
            print(f'Auditing training block {block["block"]}: 32 paired windows...', flush=True)
            x, y = read_block(block, shards, 'train', w, n, train['first_input_start_ns'], step)
            with torch.enable_grad():
                gradients = audit_batch(model, torch.from_numpy(x).to(device), torch.from_numpy(y).to(device))
            batches.append(dict(block_id=block['block'], start_index=block['start_index'],
                                end_index=block['end_index'], gradients=gradients))
            for name, value in model.state_dict().items():
                require(torch.equal(value.detach().cpu(), before[name]), f'Model state changed: {name}')
            report['completed_batches'] = len(batches)
            write_json(output/'gradient_details.json', batches)
            write_json(report_path, report)
        report['findings'] = interpretation_flags(batches)
        report['model_state_unchanged'] = True
        report['parameter_grad_buffers_remain_none'] = all(p.grad is None for p in model.parameters())
        require(report['parameter_grad_buffers_remain_none'], 'Unexpected accumulated gradients')
        for path, digest in tracked.items():
            require(sha256_file(path) == digest, f'Input changed during audit: {path}')
        report['input_sha256'] = tracked
        report['artifacts'] = {name: sha256_file(output/name) for name in ('gradient_plan.json', 'gradient_details.json')}
        report['scientific_claim_decision'] = 'local_gradient_evidence_requires_review'
        # COMPLETE denotes execution, not good gradient health. Findings remain explicit.
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
