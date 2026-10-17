"""Synthetic runner integration with mocked PyTorch/artifact loading.
Run: python tests/test_r3_primary_runner.py
Actual timestamp files, manifests, hashes, distances, and output files are used.
This does not test torch deserialization or replace Colab artifact validation.
"""
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.scientific_validation.v1.r3_pilot import CONDITIONS, LEVELS, make_block_plan
from src.scientific_validation.v1.r3_primary import configuration_hash


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    torch = types.ModuleType('torch'); torch.__version__ = 'MOCK-no-deserialization'
    sys.modules['torch'] = torch
    folders = types.ModuleType('src.utils.get_folders_utils')
    artifact_io = types.ModuleType('src.scientific_validation.v1.v1b_io')
    provenance = types.ModuleType('src.scientific_validation.v1.v1b_provenance')
    provenance.sha256_file = sha
    sys.modules[folders.__name__] = folders
    sys.modules[artifact_io.__name__] = artifact_io
    sys.modules[provenance.__name__] = provenance
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary); base = root / 'CU'; pilot_dir = root / 'pilot'
        pilot_dir.mkdir(); (root/'configs').mkdir()
        (base/'hdeg_checkpoints').mkdir(parents=True)
        checkpoint = base/'hdeg_checkpoints/epoch_0001.pt'; checkpoint.write_bytes(b'synthetic')
        cfg = dict(project_root_dir=str(root), seed=42,
                   preprocessing=dict(downsample_freq='1s'),
                   hdeg=dict(hbf=dict(checkpoint='epoch_0001.pt')), v1_validation={})
        (root/'configs/config.yaml').write_text(yaml.safe_dump(cfg))
        preflight = dict(status='PASS', checkpoint_sha256=sha(checkpoint), splits={})
        plans = {}; manifests = {}
        for k, (condition, split) in enumerate(CONDITIONS.items()):
            folder = base/'windows'/split; folder.mkdir(parents=True)
            ts = np.datetime64('2022-01-01', 'ns') + np.arange(5000)*np.timedelta64(1, 's')
            shards = []
            for sid, start in enumerate(range(0, 5000, 700)):
                end = min(start+700, 5000)
                np.savez(folder/f'shard_{sid:06d}.npz', split=split, shard_index=sid,
                         start_index=start, end_index=end, window_start_timestamps=ts[start:end])
                shards.append(dict(shard_index=sid, start_index=start, end_index=end, sample_count=end-start))
            raw = dict(num_shards=len(shards), num_samples=5000, split=split,
                       num_devices=1, window_size=30)
            source=folder/'manifest.json'; source.write_text(json.dumps(raw))
            entry=dict(shards=shards, sample_count=5000, window_manifest_sha256=sha(source))
            preflight['splits'][split]=entry; manifests[split]=entry
            plans[condition], _ = make_block_plan(ts, step_ns=10**9, block_length=128,
                  pilot_blocks=8, primary_blocks=32, seed=42+k, window_size=30)
        preflight_path=base/'v1_validation/preflight.json'; preflight_path.parent.mkdir()
        preflight_path.write_text(json.dumps(preflight))
        pilot=dict(status='COMPLETE', dataset='CU', run_id='synthetic', levels=dict.fromkeys(LEVELS),
                   configuration=cfg, configuration_sha256=configuration_hash(cfg), block_plans=plans,
                   settings=dict(block_length_windows=128, pilot_blocks_per_condition=8,
                       primary_blocks_per_condition=32, resampling_replicates=100,
                       selection_seed=42, resampling_seed=1042),
                   checkpoint_sha256=sha(checkpoint), preflight_sha256=sha(preflight_path),
                   dimensions=dict(N=1,K=1,D=1,W=30),
                   source_code_sha256={'run_r3_pilot.py':sha(ROOT/'run_r3_pilot.py'),
                   'r3_pilot.py':sha(ROOT/'src/scientific_validation/v1/r3_pilot.py')})
        (pilot_dir/'pilot_report.json').write_text(json.dumps(pilot))
        (pilot_dir/'block_plan.json').write_text(json.dumps(plans))
        folders.get_processed_folder=lambda config: base
        artifact_io.discover_shards=lambda base, split: list(range(len(manifests[split]['shards'])))
        def load(base, split, sid, *, levels, return_metadata, expected_checkpoint_sha256):
            assert expected_checkpoint_sha256 == sha(checkpoint)
            item=manifests[split]['shards'][sid]
            reps, metas = {}, {}
            for level in levels:
                a=np.arange(item['start_index'],item['end_index'],dtype=np.float32)/5000
                a += list(CONDITIONS.values()).index(split)*.2
                reps[level]=a.reshape((-1,1) if level=='g' else (-1,1,1))
                metas[level]=dict(split=split, shard_index=sid,start_index=item['start_index'],
                                 end_index=item['end_index'],num_devices=1,num_states=1,
                                 embedding_dim=1,window_size=30)
            return reps, metas
        artifact_io.load_representation_shard=load
        import run_r3_primary as runner
        old_cwd=os.getcwd(); old_argv=sys.argv
        try:
            os.chdir(root); sys.argv=['run_r3_primary.py','--pilot_dir',str(pilot_dir)]
            with contextlib.redirect_stdout(io.StringIO()): runner.main()
            outputs=list((base/'v1_validation/r3_primary').glob('*'))
            assert len(outputs)==1
            out=outputs[0]; result=json.loads((out/'primary_report.json').read_text())
            assert result['status']=='COMPLETE' and result['pilot_only'] is False
            assert result['block_plans']==plans and result['settings']['resampling_replicates']==1000
            assert not list(out.glob('selected_*.npy'))
            with np.load(out/'selected_indices.npz') as saved:
                for c in CONDITIONS:
                    expected=np.concatenate([np.arange(b['start_index'],b['end_index'])
                                              for b in plans[c]['primary_blocks']])
                    np.testing.assert_array_equal(saved[c], expected)
            for level in LEVELS:
                with np.load(out/f'resampling_values_{level}.npz') as draws:
                    assert all(draws[k].shape==(1000,) for k in draws.files)
                for r in result['levels'][level]['results'].values():
                    assert r['pilot_only'] is False and r['analysis_role']=='primary_descriptive'
                    assert r['population_ci_status']=='not_established'
            # Fail before computation when checkpoint bytes change.
            checkpoint.write_bytes(b'changed')
            try:
                with contextlib.redirect_stdout(io.StringIO()): runner.main()
            except ValueError:
                pass
            else: raise AssertionError('Changed checkpoint accepted')
            reports=[json.loads(p.read_text()) for p in
                     (base/'v1_validation/r3_primary').glob('*/primary_report.json')]
            failed=[r for r in reports if r['status']=='FAILED']
            assert len(failed)==1 and failed[0]['levels']=={}
        finally:
            os.chdir(old_cwd); sys.argv=old_argv
    print('PASS: synthetic full runner, 4 levels/3 contrasts, cross-shard selections, primary labels, outputs and checkpoint failure.')


if __name__=='__main__': main()
