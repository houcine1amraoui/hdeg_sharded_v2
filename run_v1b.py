#!/usr/bin/env python3
"""Level 1 input audit by default; --execute explicitly runs R1/R2/R3."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import yaml
from src.scientific_validation.v1.v1b_engine import V1BConfig, V1BEngine
from src.scientific_validation.v1.v1b_evidence import write_v1b_evidence
from src.scientific_validation.v1.v1b_io import load_representation_shard, save_json
from src.scientific_validation.v1.v1b_provenance import sha256_file
from src.utils.seed import set_seed
from src.utils.get_folders_utils import get_processed_folder
from src.utils.experiment_protection import assert_output_unprotected


def main():
    with open("configs/config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_root_dir", type=str)
    parser.add_argument("--execute", action="store_true",
                        help="Run scientific metrics after the input preflight.")
    args = parser.parse_args()
    if args.project_root_dir is not None:
        config["project_root_dir"] = args.project_root_dir
    seed = int(config["seed"])
    set_seed(seed)
    assert_output_unprotected(get_processed_folder(config), "run_v1b.py")
    base = Path(get_processed_folder(config))
    output = base / "v1_validation"
    try:
        settings = config["v1_validation"]
        checkpoint = config["hdeg"]["hbf"]["checkpoint"]
        if not isinstance(checkpoint, str) or not checkpoint.strip():
            raise ValueError("hdeg.hbf.checkpoint must name a checkpoint.")
        checkpoint_path = Path(checkpoint)
        if not checkpoint_path.is_absolute():
            checkpoint_path = base / "hdeg_checkpoints" / checkpoint_path
        checkpoint_hash = sha256_file(checkpoint_path)
        manifest = json.loads((base / "windows" / "val" / "manifest.json").read_text())
        reps, metadata = load_representation_shard(
            base, "val", 0, levels=("S",), return_metadata=True,
            expected_checkpoint_sha256=checkpoint_hash,
        )
        states = int(metadata["S"]["num_states"])
        dim = int(metadata["S"]["embedding_dim"])
        del reps, metadata
        cfg = V1BConfig(
            artifact_root=base, window_root=base / "windows", output_root=output,
            dataset=base.name, seed=seed, bootstrap_seed=seed,
            checkpoint_sha256=checkpoint_hash,
            window_size=int(manifest["window_size"]),
            num_devices=int(manifest["num_devices"]),
            num_states=states, representation_dim=dim,
            r2_pair_count=int(settings["r2-pairs"]),
            r3_max_per_condition=int(settings["r3-max"]),
            r1_reservoir_size=int(settings["r1-reservoir"]),
            r1_chunk_size=int(settings["r1-chunk"]),
            r3_block_length_windows=int(settings.get("r3-block-length-windows", 30)),
            bootstrap_replicates=int(settings["bootstrap-replicates"]),
        )
        engine = V1BEngine(cfg)
        preflight = engine.preflight()
        preflight["checkpoint"] = str(checkpoint_path)
        preflight["configuration"] = config
        preflight["configuration_sha256"] = hashlib.sha256(
            json.dumps(config, sort_keys=True, default=str).encode()
        ).hexdigest()
        preflight["execution_requested"] = args.execute
        preflight["r3_cost_note"] = (
            "Exact energy distance is quadratic in selected rows; 50,000 rows "
            "means billions of pair distances per contrast/level, before bootstrap."
        )
        save_json(output / "preflight.json", preflight)
    except Exception as exc:
        save_json(output / "preflight.json", {
            "status": "FAIL", "error": str(exc), "scientific_tests_executed": False,
        })
        raise SystemExit(f"Level 1 preflight failed: {exc}") from exc

    print(f"Level 1 preflight: PASS; checkpoint SHA256={checkpoint_hash}")
    print("Required populations: val, actor2_test, actor1_test (DBRL/BSE/BIL/EBRL).")
    print(f"Report: {output / 'preflight.json'}")
    if not args.execute:
        print("No scientific metrics executed. Review the preflight report first.")
        return

    print(preflight["r3_cost_note"])
    evidence = engine.run()
    evidence["preflight"] = preflight
    evidence["manifest"]["configuration"] = config
    evidence["manifest"]["checkpoint"] = str(checkpoint_path)
    evidence["manifest"]["configuration_sha256"] = preflight["configuration_sha256"]
    write_v1b_evidence(output, evidence)
    print(f"Level 1 quantitative evidence written to {output}; no claim decision performed.")


if __name__ == "__main__":
    main()
