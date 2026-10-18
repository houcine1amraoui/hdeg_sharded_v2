#!/usr/bin/env python3
"""Freeze legacy output writes and preserve a checkpoint in a new experiment directory."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import yaml
from src.utils.get_folders_utils import get_processed_folder
from src.utils.experiment_protection import (
    MARKER, sha256_file, fresh_run_directory, write_json_new, copy_checkpoint_new,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project_root_dir')
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--expected_checkpoint_sha256', required=True)
    args = parser.parse_args()
    with open('configs/config.yaml', encoding='utf-8') as stream:
        config = yaml.safe_load(stream)
    if args.project_root_dir is not None:
        config['project_root_dir'] = args.project_root_dir
    base = Path(get_processed_folder(config)).resolve()
    source = args.checkpoint or Path(config['hdeg']['hbf']['checkpoint'])
    if not source.is_absolute():
        source = base/'hdeg_checkpoints'/source
    source = source.resolve()
    expected = args.expected_checkpoint_sha256.lower()
    if len(expected) != 64 or any(c not in '0123456789abcdef' for c in expected):
        raise ValueError('Expected hash must have 64 hexadecimal characters')
    if sha256_file(source) != expected:
        raise ValueError('Checkpoint differs from pinned SHA256; no changes made')
    anchor_path = base/'v1_validation/preflight.json'
    anchor = json.loads(anchor_path.read_text())
    if anchor.get('status') != 'PASS' or not anchor.get('checkpoint_sha256'):
        raise ValueError('Expected existing PASS Level 1 preflight; no changes made')
    reference = dict(path=str(anchor_path), sha256=sha256_file(anchor_path),
                     checkpoint_sha256=anchor['checkpoint_sha256'])
    marker = base/MARKER
    protected = dict(schema_version=1, created_utc=datetime.now(timezone.utc).isoformat(),
                     level1_reference=reference,
                     protected_outputs=['dbrl', 'bse', 'bil', 'ebrl', 'hbf', 'mbai', 'v1_validation'],
                     preservation='In place; application write guard, not a backup or filesystem lock')
    if marker.exists():
        old = json.loads(marker.read_text())
        if old.get('schema_version') != 1 or old.get('level1_reference') != reference:
            raise ValueError('Existing protection marker differs from current Level 1 reference')
    else:
        write_json_new(marker, protected)
    print(f'Legacy output protection active: {marker}', flush=True)
    run = fresh_run_directory(base/'experiments')
    destination = run/'checkpoints'/'epoch_0001.pt'
    write_json_new(run/'reservation.json', dict(status='RESERVED', source_checkpoint=str(source),
                   expected_checkpoint_sha256=expected, legacy_reference=reference))
    copy_checkpoint_new(source, destination, expected)
    snapshot = json.loads(json.dumps(config))
    snapshot['hdeg']['hbf']['checkpoint'] = str(destination)
    for stage in ('dbrl','bse','bil','ebrl','hbf','mbai'):
        snapshot['hdeg'][stage]['overwrite'] = False
    with (run/'config_snapshot.yaml').open('x', encoding='utf-8') as stream:
        yaml.safe_dump(snapshot, stream, sort_keys=False)
    # Reserve a separate output namespace for later explicitly routed runners.
    (run/'artifacts').mkdir()
    (run/'validation').mkdir()
    write_json_new(run/'experiment_manifest.json', dict(status='PRESERVED',
                   checkpoint=str(destination), checkpoint_sha256=expected,
                   source_checkpoint=str(source), legacy_reference=reference,
                   is_same_checkpoint_as_level1=expected == reference['checkpoint_sha256'],
                   configuration_snapshot_sha256=sha256_file(run/'config_snapshot.yaml'),
                   output_routing='RESERVED_ONLY: stage/preflight runners are not redirected by this script',
                   original_checkpoint_recovered=False,
                   scientific_tests_executed=False))
    print(f'Preserved checkpoint: {destination}', flush=True)
    print(f'New experiment manifest: {run / "experiment_manifest.json"}', flush=True)
    print('Original config and Level 1 evidence unchanged. No training or artifact regeneration performed.')


if __name__ == '__main__':
    main()
