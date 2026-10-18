#!/usr/bin/env python3
"""Standalone, read-only Level 2 readiness audit of the frozen HDEG checkpoint."""
import argparse
from datetime import datetime, timezone
import inspect
import json
from pathlib import Path
import time
import uuid
import yaml
from src.utils.get_folders_utils import get_processed_folder
from src.scientific_validation.v2 import preflight as helper
from src.scientific_validation.v2.preflight import (
    sha256_file, require, config_differences, audit_windows,
    check_split_boundaries, inventory, load_experiment_context,
)


def write_report(path, report):
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project_root_dir')
    parser.add_argument('--experiment_manifest', type=Path,
                        help='Explicit preserved-run mode; use its checkpoint, snapshot and validation directory')
    parser.add_argument('--checkpoint', type=Path,
                        help='Defaults to hdeg.hbf.checkpoint; relative paths use hdeg_checkpoints')
    parser.add_argument('--expected_checkpoint_sha256',
                        help='Optional additional pinned hash; required if no Level 1 preflight exists')
    parser.add_argument('--v1_preflight', type=Path,
                        help='Defaults to <processed>/v1_validation/preflight.json')
    parser.add_argument('--history', action='append', type=Path, default=[],
                        help='Optional existing CSV/JSON/JSONL history; repeat for multiple files')
    args = parser.parse_args()
    if args.experiment_manifest is not None and (args.checkpoint is not None or args.v1_preflight is not None):
        parser.error('--experiment_manifest cannot be combined with --checkpoint or --v1_preflight')
    # Preserve the established configuration style and processed-path helper.
    with open('configs/config.yaml', encoding='utf-8') as stream:
        config = yaml.safe_load(stream)
    require(isinstance(config, dict), 'config.yaml must contain a mapping')
    if args.project_root_dir is not None:
        config['project_root_dir'] = args.project_root_dir
    context = None
    if args.experiment_manifest is not None:
        context = load_experiment_context(args.experiment_manifest, config, get_processed_folder)
        config = context['config']
    base = Path(get_processed_folder(config)).resolve()
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8]
    # Always a fresh Level 2 directory; no user path can redirect output over Level 1.
    output = (context['output_parent'] if context else base/'v2_validation'/'preflight')/run_id
    output.mkdir(parents=True, exist_ok=False)
    path = output/'preflight_report.json'
    started = time.perf_counter()
    report = dict(status='RUNNING', run_id=run_id, configuration=config,
                  scientific_tests_executed=False, optimizer_steps=0,
                  scientific_claim_decision='not_performed', splits={}, warnings=[],
                  audit_scope='Frozen checkpoint and train/val readiness; no test-split evaluation',
                  source_code_sha256={Path(__file__).name: sha256_file(__file__),
                                      'v2/preflight.py': sha256_file(helper.__file__)})
    report['mode'] = 'new_experiment' if context else 'level1_linked'
    if context:
        report['experiment'] = context['provenance']
        report['warnings'].append('Historical Level 1 results do not validate this experiment; no Level 1 evidence is inherited')
    write_report(path, report)
    print(f'Level 2 preflight report: {path}', flush=True)
    try:
        checkpoint = context['checkpoint'] if context else (args.checkpoint or Path(config['hdeg']['hbf']['checkpoint']))
        if not checkpoint.is_absolute():
            checkpoint = base/'hdeg_checkpoints'/checkpoint
        checkpoint = checkpoint.resolve()
        digest = sha256_file(checkpoint)
        report['checkpoint'] = dict(path=str(checkpoint), sha256=digest)
        anchor_path = args.v1_preflight or base/'v1_validation'/'preflight.json'
        anchor = None
        if context:
            # Verified by the manifest helper solely as historical provenance.
            # Do not impose its checkpoint or validation-manifest hashes on this new run.
            pass
        elif anchor_path.is_file():
            anchor = json.loads(anchor_path.read_text())
            require(anchor.get('status') == 'PASS', 'Level 1 preflight is not PASS')
            require(anchor.get('checkpoint_sha256') == digest, 'Checkpoint differs from Level 1')
            report['level1_reference'] = dict(path=str(anchor_path.resolve()),
                                             sha256=sha256_file(anchor_path))
        else:
            require(args.v1_preflight is None, 'Explicit Level 1 preflight path does not exist')
            require(args.expected_checkpoint_sha256 is not None,
                    'Provide --expected_checkpoint_sha256 when Level 1 preflight is unavailable')
            report['warnings'].append('Level 1 reference absent; only the explicitly pinned checkpoint hash is checked')
        if args.expected_checkpoint_sha256 is not None:
            expected = args.expected_checkpoint_sha256.lower()
            require(len(expected) == 64 and all(c in '0123456789abcdef' for c in expected),
                    'Expected checkpoint hash must be 64 hexadecimal characters')
            require(digest == expected, 'Checkpoint differs from explicitly pinned SHA256')
        write_report(path, report)
        # Delayed imports ensure a dependency failure is recorded in this run's report.
        import torch
        from src.training.train_hdeg_e2e import build_live_model
        from src.common.graph.semantics import load_behavioral_state_config
        from run_dbrl_sharded import load_manifest, resolve_shard_path
        print('Loading the hash-verified project checkpoint on CPU...', flush=True)
        # Same trusted-project-checkpoint loading convention as existing stage runners.
        payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
        require(isinstance(payload, dict), 'Checkpoint payload must be a dictionary')
        state, saved = payload.get('model_state_dict'), payload.get('config')
        require(isinstance(state, dict) and isinstance(saved, dict), 'Missing state_dict or saved config')
        epoch = payload.get('epoch')
        require(isinstance(epoch, int) and not isinstance(epoch, bool) and epoch >= 0, 'Invalid checkpoint epoch')
        differences = config_differences(config, saved)
        report['configuration_differences'] = differences
        require(not any(d['category'] == 'experiment_critical' for d in differences),
                'Experiment-critical checkpoint/config mismatch; inspect configuration_differences')
        report['checkpoint'].update(epoch=epoch, saved_configuration=saved,
                                    optimizer_state_present='optimizer_state_dict' in payload)
        metrics = payload.get('metrics')
        # Do not interpret training metrics as validation history.
        report['checkpoint']['training_metrics'] = (
            {k: v for k, v in metrics.items() if isinstance(v, (int, float, str, bool)) or v is None}
            if isinstance(metrics, dict) else None)
        # Reject invalid scalar metrics instead of emitting non-standard JSON.
        json.dumps(report, allow_nan=False)
        semantics_path = Path(config['project_root_dir'])/'configs'/'hdeg'/'behavioral_states.yaml'
        devices_path = base/'devices.json'
        semantics = load_behavioral_state_config(config_path=semantics_path, devices_path=devices_path)
        mask = semantics.torch_mask(dtype=torch.float32, device=torch.device('cpu'), clone=True)
        model = build_live_model(config=config, behavioral_config=semantics,
                                 compatibility_mask=mask, device=torch.device('cpu'))
        # Compare fixed buffers BEFORE load_state_dict overwrites them, including BIL topology/MO weights.
        for name, value in model.named_buffers():
            require(name in state and isinstance(state[name], torch.Tensor), f'Missing buffer: {name}')
            require(value.dtype == state[name].dtype and torch.equal(value, state[name]),
                    f'Fixed buffer differs from current semantics/configuration: {name}')
        for name, value in state.items():
            require(isinstance(value, torch.Tensor), f'Non-tensor state entry: {name}')
            require(torch.isfinite(value).all().item(), f'Nonfinite checkpoint tensor: {name}')
        expected_state = model.state_dict()
        require(set(state) == set(expected_state), 'Unexpected/missing model state keys')
        for name, value in expected_state.items():
            require(value.shape == state[name].shape and value.dtype == state[name].dtype,
                    f'State shape/dtype mismatch: {name}')
        model.load_state_dict(state, strict=True)
        model.eval()
        report['checkpoint']['strict_model_load'] = 'PASS'
        report['semantics'] = dict(devices_sha256=sha256_file(devices_path),
                                   behavioral_states_sha256=sha256_file(semantics_path),
                                   device_ids=list(semantics.device_ids()),
                                   state_ids=list(semantics.state_ids()),
                                   fixed_buffer_comparison='PASS',
                                   training_time_device_order='not_recorded_in_current_checkpoint_schema')
        report['warnings'].append(
            'Current device order and checkpoint mask agree, but the checkpoint does not store original device IDs; '
            'permutations within identical compatibility-mask columns cannot be independently excluded')
        report['dimensions'] = dict(N=semantics.num_devices, K=semantics.num_states,
                                    D=config['hdeg']['dbrl']['embedding_dim'],
                                    W=config['preprocessing']['window_size'])
        # Hash all directly constructed model classes, plus shared readers and builders.
        for obj in (build_live_model, load_behavioral_state_config, load_manifest,
                    get_processed_folder, *(type(m) for m in model.children())):
            source = inspect.getsourcefile(obj)
            if source:
                report['source_code_sha256'][str(Path(source).resolve())] = sha256_file(source)
        del model, expected_state, state, payload
        report['evidence_inventory'] = inventory(checkpoint.parent, args.history)
        write_report(path, report)
        for split in ('train', 'val'):
            checked = audit_windows(base, split, config, load_manifest, resolve_shard_path)
            report['splits'][split] = checked
            require(checked['num_devices'] == semantics.num_devices, 'Manifest/semantic device-count mismatch')
            if split == 'val' and anchor is not None:
                require(checked['manifest_sha256'] == anchor['splits']['val']['window_manifest_sha256'],
                        'Validation manifest differs from Level 1 reference')
            write_report(path, report)
        report['split_boundaries'] = check_split_boundaries(report['splits']['train'], report['splits']['val'])
        require(sha256_file(checkpoint) == digest, 'Checkpoint changed during preflight')
        if anchor is not None:
            require(sha256_file(anchor_path) == report['level1_reference']['sha256'],
                    'Level 1 reference changed during preflight')
        if context:
            for source, expected_hash in context['tracked_hashes'].items():
                require(sha256_file(source) == expected_hash, 'Experiment input changed during preflight: '+source)
        report.update(status='COMPLETE', readiness=dict(
            bounded_frozen_checkpoint_diagnostic='READY_WITH_REPORTED_LIMITATIONS',
            convergence='NOT_ESTABLISHED', best_validation_selection='NOT_ESTABLISHED',
            exact_training_resume='NOT_ASSESSED', new_training_run='SAFEGUARDS_REQUIRED'))
        report['warnings'].extend([
            'Tensor/target-timestamp checks cover first/middle/last shards only; unsampled tensor corruption is not excluded',
            'Scaler fitting history and original raw-data provenance are not independently reconstructed',
            'No forecasts, persistence comparisons, learning curves, or scientific claims are evaluated by this preflight'])
        print('Preflight complete: ready for a bounded diagnostic, subject to the reported limitations.', flush=True)
    except Exception as exc:
        report.update(status='FAIL', error_type=type(exc).__name__, error=str(exc),
                      readiness=dict(bounded_frozen_checkpoint_diagnostic='BLOCKED'))
        # A malformed checkpoint metric may itself have caused JSON rejection.
        if 'checkpoint' in report:
            report['checkpoint'].pop('training_metrics', None)
        raise
    finally:
        report['elapsed_seconds'] = time.perf_counter()-started
        # Sanitize nonfinite values only on failure so the failure itself is always reportable.
        if report['status'] == 'FAIL':
            import math
            def clean(value):
                if isinstance(value, float) and not math.isfinite(value):
                    return str(value)
                if isinstance(value, dict):
                    return {str(k): clean(v) for k, v in value.items()}
                if isinstance(value, (list, tuple)):
                    return [clean(v) for v in value]
                return value
            report = clean(report)
        write_report(path, report)
        print(f'Report saved: {path}', flush=True)


if __name__ == '__main__':
    main()
