#!/usr/bin/env python3
"""Standalone R3 block-length sensitivity: 64 or 256 windows, 32 blocks, 1000 resamples."""
from __future__ import annotations
import argparse
import contextlib
from datetime import datetime, timezone
import gc
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import sys
import time
import uuid
import numpy as np
import torch
import yaml
from src.utils.get_folders_utils import get_processed_folder
from src.scientific_validation.v1.v1b_io import discover_shards, load_representation_shard
from src.scientific_validation.v1.v1b_provenance import sha256_file
from src.scientific_validation.v1 import r3_pilot as helper
from src.scientific_validation.v1.r3_pilot import (
    CONDITIONS, LEVELS, CONTRASTS, positive_int,
    block_distance_means, make_resampling_weights, summarize_resampling,
)


from run_r3_pilot import (write_report, peak_rss_mib, sampling_step_ns,
                          read_timestamps, collect_level)
from src.scientific_validation.v1 import r3_primary as primary_helper
from src.scientific_validation.v1.r3_primary import frozen_primary_indices
from src.scientific_validation.v1 import r3_sensitivity as sensitivity_helper
from src.scientific_validation.v1.r3_sensitivity import validate_primary, sensitivity_plan
from src.scientific_validation.v1.r3_pilot import index_hash


def main():
    with open("configs/config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project_root_dir")
    parser.add_argument("--primary_dir", required=True, type=Path,
                        help="Completed primary directory containing primary_report.json and block_plan.json")
    parser.add_argument("--block_length", required=True, type=int, choices=(64, 256))
    args = parser.parse_args()
    if args.project_root_dir is not None:
        config["project_root_dir"] = args.project_root_dir
    base = Path(get_processed_folder(config))
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
    output = base / "v1_validation" / "r3_sensitivity" / f"L{args.block_length}" / run_id
    output.mkdir(parents=True)
    report_path = output / "sensitivity_report.json"
    started = time.perf_counter()
    report = {"status": "RUNNING", "run_id": run_id, "dataset": base.name,
              "purpose": "R3 descriptive block-length sensitivity analysis", "pilot_only": False, "population_ci_status": "not_established",
              "configuration": config, "levels": {}, "timing_seconds": {},
              "memory_measure": "process lifetime peak RSS; includes imported libraries and input shards"}
    write_report(report_path, report)
    print(f"R3 sensitivity output: {output}", flush=True)
    try:
        settings = config["v1_validation"].get("r3-sensitivity", {})
        if not isinstance(settings, dict) or set(settings) - {"tile_size"}:
            raise ValueError("v1_validation.r3-sensitivity permits only tile_size")
        tile = positive_int(settings.get("tile_size", 128), "tile_size")
        if tile > 512:
            raise ValueError("tile_size must not exceed 512")
        L, B, R = args.block_length, 32, 1000
        seed = config["seed"]
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        primary_path = args.primary_dir / "primary_report.json"
        plan_path = args.primary_dir / "block_plan.json"
        primary = json.loads(primary_path.read_text(encoding="utf-8"))
        frozen_plans = json.loads(plan_path.read_text(encoding="utf-8"))
        validate_primary(primary, frozen_plans, config, base.name)
        report["settings"] = dict(block_length_windows=L, blocks_per_condition=B,
                                  selected_rows_per_condition=B * L, resampling_replicates=R,
                                  tile_size=tile, selection_seed=seed, resampling_seed=seed + 1000,
                                  unrelated_main_r3_settings_used=False)
        report["primary_provenance"] = dict(run_id=primary["run_id"], directory=str(args.primary_dir),
                                          report_sha256=sha256_file(primary_path),
                                          block_plan_sha256=sha256_file(plan_path))
        report["sensitivity_design"] = {
            "primary_block_length": 128, "evaluated_block_length": L,
            "fixed_block_count": 32, "sample_size_changes_with_block_length": True,
            "selection": "Partition each original time segment at the requested length; draw 32 blocks without replacement using seed + condition offset. No metric-dependent selection.",
            "comparison_limit": "Block length, sampled windows and sample size change together. This is not an isolated block-length effect or an independent replication.",
            "resampling_alignment": "Same seeded block-position weights across runs; block positions represent different time ranges at different lengths."}
        report["uncertainty_scope"] = "conditional_selected_block_resampling"
        report["interpretation"] = (
            "q025/q975 are descriptive conditional resampling quantiles, not calibrated population "
            "confidence intervals. No p-values or scientific pass/fail decisions. Conditions are "
            "resampled independently; cross-period covariance and training uncertainty are not modeled.")
        checkpoint = config["hdeg"]["hbf"]["checkpoint"]
        if not isinstance(checkpoint, str) or not checkpoint.strip():
            raise ValueError("hdeg.hbf.checkpoint must name the trained checkpoint")
        checkpoint_path = Path(checkpoint)
        if not checkpoint_path.is_absolute():
            checkpoint_path = base / "hdeg_checkpoints" / checkpoint_path
        checkpoint_hash = sha256_file(checkpoint_path)
        preflight_path = base / "v1_validation" / "preflight.json"
        preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
        if preflight.get("status") != "PASS" or preflight.get("checkpoint_sha256") != checkpoint_hash:
            raise ValueError("A passing Level 1 preflight for this checkpoint is required")
        if checkpoint_hash != primary["checkpoint_sha256"]:
            raise ValueError("Checkpoint changed since the completed primary")
        if sha256_file(preflight_path) != primary["preflight_sha256"]:
            raise ValueError("Preflight changed since the completed primary")
        # Retain the primary run's source dependencies exactly.
        import run_r3_pilot as pilot_runner
        import run_r3_primary as primary_runner
        for name, path in (("run_r3_primary.py", Path(primary_runner.__file__)),
                           ("r3_primary.py", Path(primary_helper.__file__)),
                           ("run_r3_pilot.py", Path(pilot_runner.__file__)),
                           ("r3_pilot.py", Path(helper.__file__))):
            if sha256_file(path) != primary["source_code_sha256"].get(name):
                raise ValueError(f"Primary dependency changed: {name}")
        report["checkpoint"] = str(checkpoint_path)
        report["checkpoint_sha256"] = checkpoint_hash
        report["preflight_sha256"] = sha256_file(preflight_path)
        report["source_code_sha256"] = {
            "run_r3_sensitivity.py": sha256_file(Path(__file__)),
            "r3_sensitivity.py": sha256_file(Path(sensitivity_helper.__file__)),
            "run_r3_primary.py": sha256_file(Path(primary_runner.__file__)),
            "r3_primary.py": sha256_file(Path(primary_helper.__file__)),
            "run_r3_pilot.py": sha256_file(Path(pilot_runner.__file__)),
            "r3_pilot.py": sha256_file(Path(helper.__file__)),
        }
        report["configuration_sha256"] = hashlib.sha256(
            json.dumps(config, sort_keys=True).encode()).hexdigest()
        step = sampling_step_ns(config["preprocessing"]["downsample_freq"])
        raw_manifests, plans, selected, manifests = {}, {}, {}, {}
        for k, (condition, split) in enumerate(CONDITIONS.items()):
            print(f"Checking timestamps against frozen blocks: {condition} / {split}", flush=True)
            entry = preflight["splits"][split]
            source = base / "windows" / split / "manifest.json"
            if sha256_file(source) != entry["window_manifest_sha256"]:
                raise ValueError(f"Window manifest changed since preflight: {split}")
            raw = json.loads(source.read_text())
            ids = discover_shards(base, split)
            if len(ids) != len(entry["shards"]) or len(ids) != raw["num_shards"]:
                raise ValueError(f"Shard count changed since preflight: {split}")
            if raw["num_samples"] != entry["sample_count"] or raw["split"] != split:
                raise ValueError(f"Window population mismatch: {split}")
            actual_windows = sorted(p.name for p in (base / "windows" / split).glob("shard_*.npz"))
            if actual_windows != [f"shard_{i:06d}.npz" for i in ids]:
                raise ValueError(f"Window shard set mismatch: {split}")
            timestamps = read_timestamps(base, split, entry)
            original_indices = frozen_primary_indices(
                frozen_plans[condition], timestamps, step_ns=step, seed=seed + k,
                window_size=int(raw["window_size"]))
            if index_hash(original_indices) != primary["primary_index_sha256"][condition]:
                raise ValueError(f"Primary index hash mismatch: {condition}")
            plans[condition], selected[condition] = sensitivity_plan(
                timestamps, length=L, step_ns=step, seed=seed + k,
                window_size=int(raw["window_size"]))
            raw_manifests[condition], manifests[condition] = raw, entry
            del timestamps
        report["block_plans"] = plans
        report["selected_index_sha256"] = {c: index_hash(v) for c, v in selected.items()}
        write_report(output / "block_plan.json", plans)
        np.savez(output / "selected_indices.npz", **selected)
        first, meta = load_representation_shard(base, "val", 0, levels=("S",),
                                               return_metadata=True, expected_checkpoint_sha256=checkpoint_hash)
        dimensions = (int(raw_manifests["N1"]["num_devices"]), int(meta["S"]["num_states"]),
                      int(meta["S"]["embedding_dim"]), int(raw_manifests["N1"]["window_size"]))
        del first, meta
        for raw in raw_manifests.values():
            if (raw["num_devices"], raw["window_size"]) != (dimensions[0], dimensions[3]):
                raise ValueError("Window dimensions differ across conditions")
        report["dimensions"] = dict(zip(("N", "K", "D", "W"), dimensions))
        if report["dimensions"] != primary["dimensions"]:
            raise ValueError("Representation dimensions changed since primary")
        weights = {condition: make_resampling_weights(B, R, seed + 1000 + k)
                   for k, condition in enumerate(CONDITIONS)}
        np.savez(output / "resampling_weights.npz", **weights)
        report["timing_seconds"]["input_validation_and_planning"] = time.perf_counter() - started
        blas = io.StringIO()
        with contextlib.redirect_stdout(blas):
            np.show_config()
        report["environment"] = {"python": sys.version, "numpy": np.__version__,
                                 "torch": torch.__version__, "platform": platform.platform(),
                                 "cpu_count": os.cpu_count(), "numpy_configuration": blas.getvalue(),
                                 "thread_environment": {k: os.environ.get(k) for k in
                                    ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}}
        try:
            from threadpoolctl import threadpool_info
            report["environment"]["threadpools"] = threadpool_info()
        except ImportError:
            report["environment"]["threadpools"] = "threadpoolctl unavailable"
        write_report(report_path, report)
        for level in LEVELS:
            print(f"Computing sensitivity level {level}", flush=True)
            t_load = time.perf_counter()
            array_paths = {}
            for condition, split in CONDITIONS.items():
                rows = collect_level(base, split, level, selected[condition], manifests[condition],
                                     checkpoint_hash, dimensions)
                path = output / f"selected_{level}_{condition}.npy"
                np.save(path, rows.reshape(B, L, -1))
                array_paths[condition] = path
                del rows
                gc.collect()
            loading_seconds = time.perf_counter() - t_load
            tables, table_times = {}, {}
            pairs = [(c, c) for c in CONDITIONS] + list(CONTRASTS)
            for a, b in pairs:
                t = time.perf_counter()
                x = np.load(array_paths[a], mmap_mode="r")
                y = x if a == b else np.load(array_paths[b], mmap_mode="r")
                last_print = [0.0]
                def progress(done, total):
                    now = time.perf_counter()
                    if now - last_print[0] >= 15 or done == total:
                        print(f"  {level} {a}-{b}: block row {done}/{total}", flush=True)
                        last_print[0] = now
                tables[(a, b)] = block_distance_means(x, y, tile_size=tile,
                                                     same=a == b, progress=progress)
                table_times[f"{a}-{b}"] = time.perf_counter() - t
                del x, y
                gc.collect()
            np.savez(output / f"block_distances_{level}.npz",
                     **{f"{a}__{b}": value for (a, b), value in tables.items()})
            t = time.perf_counter()
            results, draws = {}, {}
            for a, b in CONTRASTS:
                results[f"{a}-{b}"], draws[f"{a}__{b}"] = summarize_resampling(
                    tables[(a, a)], tables[(a, b)], tables[(b, b)], weights[a], weights[b])
                results[f"{a}-{b}"]["pilot_only"] = False
                results[f"{a}-{b}"]["analysis_role"] = "block_length_sensitivity"
            resampling_seconds = time.perf_counter() - t
            np.savez(output / f"resampling_values_{level}.npz", **draws)
            report["levels"][level] = {"results": results, "loading_seconds": loading_seconds,
                                      "distance_seconds": sum(table_times.values()),
                                      "distance_table_seconds": table_times,
                                      "resampling_seconds": resampling_seconds}
            report["peak_rss_mib"] = peak_rss_mib()
            report["timing_seconds"]["elapsed"] = time.perf_counter() - started
            write_report(report_path, report)
            # Retain summaries, selections and weights; remove only regenerable
            # arrays created by this unique sensitivity run after successful level output.
            for path in array_paths.values():
                path.unlink()
            del tables
            gc.collect()
        report["status"] = "COMPLETE"
        report["scientific_claim_decision"] = "not_performed"
    except BaseException as exc:
        report["status"] = "INTERRUPTED" if isinstance(exc, KeyboardInterrupt) else "FAILED"
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        report["peak_rss_mib"] = peak_rss_mib()
        report["timing_seconds"]["elapsed"] = time.perf_counter() - started
        write_report(report_path, report)
        print(f"R3 sensitivity {report['status']}: {report_path}", flush=True)


if __name__ == "__main__":
    main()
