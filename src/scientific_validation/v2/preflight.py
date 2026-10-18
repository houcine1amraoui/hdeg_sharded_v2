"""Read-only source checks; no training, prediction, or Level 1 writes."""
from pathlib import Path
import csv
import hashlib
import json
import re
import numpy as np


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def step_ns(value):
    match = re.fullmatch(r'\s*(\d+)\s*(ns|us|ms|s|min|h|D)\s*', str(value))
    require(match is not None, 'downsample_freq must be a fixed duration, e.g. 1s')
    units = dict(ns=1, us=1000, ms=10**6, s=10**9, min=60*10**9,
                 h=3600*10**9, D=86400*10**9)
    step = int(match[1]) * units[match[2]]
    require(step > 0, 'Timestamp step must be positive')
    return step


def config_differences(current, saved):
    """Report all differences; block changes affecting the existing experiment."""
    def flatten(value, prefix=''):
        if isinstance(value, dict):
            out = {}
            for key, item in value.items():
                out.update(flatten(item, f'{prefix}.{key}' if prefix else key))
            return out
        return {prefix: value}
    a, b = flatten(current), flatten(saved)
    path_keys = {'project_root_dir', 'preprocessing.dataset_folder'}
    architectural = {
        'hdeg.dbrl.' + k for k in ('hidden_dim', 'embedding_dim', 'gru_layers',
        'graph_top_k', 'graph_self_loops', 'graph_symmetric', 'graph_heads')}
    architectural |= {'hdeg.bse.num_heads', 'hdeg.bil.num_heads',
                      'hdeg.hbf.dynamics_hidden_dim'}
    # Unit MO weights are implicit in the current training implementation.
    for key in ('lambda_z', 'lambda_s', 'lambda_s_tilde', 'lambda_g'):
        full = 'hdeg.mo.' + key
        a.setdefault(full, 1.0)
        b.setdefault(full, 1.0)
        architectural.add(full)
    result = []
    for key in sorted(a.keys() | b.keys()):
        if key in a and key in b and a[key] == b[key]:
            continue
        category = ('path_only' if key in path_keys else
                    'experiment_critical' if key in architectural or
                    key.startswith('preprocessing.') else 'execution_or_other')
        result.append(dict(key=key, current=a.get(key), saved=b.get(key),
                           current_present=key in a, saved_present=key in b,
                           category=category))
    return result


def validate_pair_arrays(x, y, starts, targets, step):
    require(x.dtype == np.float32 and y.dtype == np.float32, 'X/Y must be float32')
    require(x.ndim == 3 and x.shape == y.shape and x.shape[0] > 0, 'Invalid X/Y shapes')
    require(np.isfinite(x).all() and np.isfinite(y).all(), 'Nonfinite X/Y values')
    require(np.array_equal(x[:, 1:], y[:, :-1]), 'Within-pair X/Y shift mismatch')
    if len(x) > 1:
        require(np.array_equal(y[:-1], x[1:]), 'Adjacent-pair X/Y shift mismatch')
    require(targets.dtype.kind == 'M' and targets.shape == x.shape[:2],
            'Invalid target timestamps')
    require(not np.isnat(targets).any(), 'NaT target timestamp')
    ticks = targets.astype('datetime64[ns]').astype(np.int64)
    require(np.array_equal(ticks, starts[:, None] +
                          step*np.arange(1, x.shape[1]+1)), 'Target timestamp shift mismatch')


def audit_windows(base, split, config, manifest_loader, shard_resolver, progress=print):
    """All manifest entries/start timestamps; first/middle/last full tensor shards."""
    directory = Path(base)/'windows'/split
    manifest_path = directory/'manifest.json'
    manifest_hash = sha256_file(manifest_path)
    m = manifest_loader(windows_dir=directory.parent, split=split)
    require(m['window_size'] == config['preprocessing']['window_size'], 'Window-size mismatch')
    count = len(m['shards'])
    require(count > 0, 'Empty manifest')
    selected = sorted({0, count//2, count-1})
    step = step_ns(config['preprocessing']['downsample_freq'])
    rows = 0
    first = previous = None
    timestamp_hash = hashlib.sha256()
    entries = []
    paths = [Path(shard_resolver(directory, entry)).resolve() for entry in m['shards']]
    require(len(set(paths)) == count, 'Duplicate manifest shard paths')
    require(all(p.parent == directory.resolve() for p in paths), 'Shard outside its split directory')
    require(set(directory.resolve().glob('shard_*.npz')) == set(paths), 'Unlisted or missing shard files')
    for sid, path in enumerate(paths):
        expected_count = min(m['shard_size'], m['num_samples']-rows)
        require(expected_count > 0, 'Invalid shard coverage')
        item = dict(shard_index=sid, file=str(path), start_index=rows,
                    end_index=rows+expected_count, tensor_values_checked=sid in selected)
        if sid in selected:
            item['sha256'] = sha256_file(path)
        with np.load(path, allow_pickle=False) as data:
            for key, value in dict(split=split, shard_index=sid, start_index=rows,
                                   end_index=rows+expected_count, window_size=m['window_size'],
                                   num_devices=m['num_devices']).items():
                require(data[key].item() == value, f'{path.name}: {key} mismatch')
            dates = data['window_start_timestamps']
            require(dates.dtype.kind == 'M' and dates.shape == (expected_count,), 'Invalid starts')
            require(not np.isnat(dates).any(), 'NaT start timestamp')
            ticks = dates.astype('datetime64[ns]').astype(np.int64)
            require(np.all(np.diff(ticks) == step), 'Irregular within-shard start timestamps')
            if previous is not None:
                require(int(ticks[0])-previous == step, 'Irregular cross-shard starts')
            if first is None:
                first = int(ticks[0])
            previous = int(ticks[-1])
            timestamp_hash.update(ticks.astype('<i8').tobytes())
            if sid in selected:
                x, y = data['X'], data['Y']
                require(x.shape == (expected_count, m['window_size'], m['num_devices']),
                        'Tensor shape/manifest mismatch')
                validate_pair_arrays(x, y, ticks, data['target_timestamps'], step)
                del x, y
        if sid in selected:
            require(sha256_file(path) == item['sha256'], 'Shard changed during audit')
        rows += expected_count
        entries.append(item)
        if sid % 10 == 0 or sid == count-1:
            progress(f'  {split}: inspected {sid+1}/{count} shard metadata/timestamps', flush=True)
    require(rows == m['num_samples'], 'Total sample coverage mismatch')
    require(sha256_file(manifest_path) == manifest_hash, 'Manifest changed during audit')
    return dict(status='PASS', manifest_sha256=manifest_hash, sample_count=rows,
                num_devices=m['num_devices'], window_size=m['window_size'],
                first_input_start_ns=first, last_input_start_ns=previous,
                last_target_end_ns=previous + m['window_size']*step,
                start_timestamps_sha256=timestamp_hash.hexdigest(),
                sampled_tensor_shards=selected, shards=entries,
                coverage=dict(manifests='all entries', start_timestamps='all train/val rows',
                              tensor_values='first/middle/last shards only',
                              target_timestamps='first/middle/last shards only',
                              tensor_content_hashes='first/middle/last shards only'))


def check_split_boundaries(train, val):
    require(train['num_devices'] == val['num_devices'], 'Train/val channel-count mismatch')
    require(train['last_target_end_ns'] < val['first_input_start_ns'],
            'Train target observations overlap or follow validation inputs')
    return dict(status='PASS', scope='All start timestamps; regular-step target extent',
                train_last_target_end_ns=train['last_target_end_ns'],
                val_first_input_start_ns=val['first_input_start_ns'])


def inventory(checkpoint_directory, history_paths):
    """Presence is not verified learning evidence; never deserialize extra checkpoints."""
    candidates = []
    for path in sorted(set(Path(checkpoint_directory).glob('*.pt')) |
                       set(Path(checkpoint_directory).glob('*.pth'))):
        match = re.fullmatch(r'epoch_(\d+)\.pt', path.name)
        candidates.append(dict(path=str(path.resolve()), bytes=path.stat().st_size,
                               epoch_from_filename=int(match[1]) if match else None,
                               payload_verified=False))
    histories = []
    for source in history_paths:
        path = Path(source).resolve()
        item = dict(path=str(path), interpretation='Unvalidated candidate history, not proof of convergence')
        try:
            item.update(sha256=sha256_file(path), bytes=path.stat().st_size)
            if path.suffix.lower() == '.csv':
                with path.open(newline='', encoding='utf-8-sig') as stream:
                    reader = csv.DictReader(stream)
                    item['fields'] = reader.fieldnames
                    item['record_count'] = sum(1 for _ in reader)
            elif path.suffix.lower() == '.jsonl':
                fields, count = set(), 0
                with path.open(encoding='utf-8') as stream:
                    for line in stream:
                        if line.strip():
                            record = json.loads(line)
                            require(isinstance(record, dict), 'JSONL record must be an object')
                            fields.update(record)
                            count += 1
                item.update(fields=sorted(fields), record_count=count)
            elif path.suffix.lower() == '.json':
                obj = json.loads(path.read_text())
                item.update(root_type=type(obj).__name__,
                            top_level_keys=sorted(obj) if isinstance(obj, dict) else None,
                            record_count=len(obj) if isinstance(obj, list) else None)
            else:
                item['format_note'] = 'File inventoried; no history parser for this format'
            item['status'] = 'PRESENT_UNVALIDATED'
        except Exception as exc:
            item.update(status='UNREADABLE', error=str(exc))
        histories.append(item)
    return dict(checkpoints=candidates, histories=histories,
                history_search_scope='Only explicitly supplied --history files',
                learning_curve_evidence='unavailable' if not histories else 'requires_content_and_provenance_review',
                convergence='not_established', validation_selection='not_established')
