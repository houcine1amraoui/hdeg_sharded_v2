#!/usr/bin/env python3
"""Standalone R3 computational pilot; never calls R1/R2 or the main engine."""
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
import re
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
    CONDITIONS, LEVELS, CONTRASTS, positive_int, make_block_plan,
    block_distance_means, make_resampling_weights, summarize_resampling,
)


def write_report(path, report):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def peak_rss_mib():
    try:
        import resource
        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return float(value / (1024 ** 2 if sys.platform == "darwin" else 1024))
    except ImportError:
        return None


def sampling_step_ns(text):
    # Fixed durations only; reject calendar offsets and ambiguous numeric dates.
    match = re.fullmatch(r"\s*(\d+)\s*(ns|us|ms|s|min|h|D)\s*", str(text))
    if not match:
        raise ValueError("downsample_freq must be a fixed duration, e.g. 1s or 3s")
    units = {"ns": 1, "us": 1000, "ms": 10**6, "s": 10**9,
             "min": 60 * 10**9, "h": 3600 * 10**9, "D": 86400 * 10**9}
    return positive_int(int(match[1]) * units[match[2]], "timestamp step")


def read_timestamps(base, split, index_manifest):
    chunks, expected = [], 0
    for sid, item in enumerate(index_manifest["shards"]):
        if (item["shard_index"] != sid or item["start_index"] != expected
                or item["end_index"] <= expected
                or item["sample_count"] != item["end_index"] - expected):
            raise ValueError(f"Invalid preflight range for {split}/{sid}")
        path = base / "windows" / split / f"shard_{sid:06d}.npz"
        with np.load(path, allow_pickle=False) as archive:
            for field, value in (("split", split), ("shard_index", sid),
                                 ("start_index", expected), ("end_index", item["end_index"])):
                if archive[field].item() != value:
                    raise ValueError(f"{path}: {field} mismatch")
            ts = archive["window_start_timestamps"]
            if ts.shape != (item["sample_count"],):
                raise ValueError(f"{path}: window timestamp count mismatch")
            if ts.dtype.kind not in "MUS":
                raise ValueError(f"{path}: unsupported timestamp dtype {ts.dtype}")
            chunks.append(ts.astype("datetime64[ns]"))
        expected = item["end_index"]
    if expected != index_manifest["sample_count"]:
        raise ValueError(f"{split}: incomplete preflight coverage")
    return np.concatenate(chunks)


def collect_level(base, split, level, indices, manifest, checkpoint_hash, dimensions):
    N, K, D, W = dimensions
    tail = {"Z": (N, D), "S": (K, D), "S_tilde": (K, D), "g": (D,)}[level]
    out = np.empty((len(indices),) + tail, dtype=np.float32)
    assigned = np.zeros(len(indices), dtype=bool)
    for item in manifest["shards"]:
        start, end, sid = item["start_index"], item["end_index"], item["shard_index"]
        positions = np.flatnonzero((indices >= start) & (indices < end))
        if not len(positions):
            continue
        reps, metadata = load_representation_shard(
            base, split, sid, levels=(level,), return_metadata=True,
            expected_checkpoint_sha256=checkpoint_hash,
        )
        a, meta = reps[level], metadata[level]
        expected_meta = dict(split=split, shard_index=sid, start_index=start,
                             end_index=end, num_devices=N, window_size=W, embedding_dim=D)
        if level != "Z":
            expected_meta["num_states"] = K
        for field, value in expected_meta.items():
            if meta.get(field) != value:
                raise ValueError(f"{split}/{sid}/{level}: {field} mismatch")
        if a.shape != (end - start,) + tail or a.dtype != np.float32:
            raise ValueError(f"{split}/{sid}/{level}: tensor shape/dtype mismatch")
        if not np.isfinite(a).all():
            raise ValueError(f"{split}/{sid}/{level}: nonfinite representations")
        out[positions] = a[indices[positions] - start]
        assigned[positions] = True
        del reps, metadata, a, meta
    if not assigned.all():
        raise ValueError(f"Unassigned selected rows for {split}/{level}")
    return out


def main():
    with open("configs/config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project_root_dir")
    args = parser.parse_args()
    if args.project_root_dir is not None:
        config["project_root_dir"] = args.project_root_dir
    base = Path(get_processed_folder(config))
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
    output = base / "v1_validation" / "r3_pilot" / run_id
    output.mkdir(parents=True)
    report_path = output / "pilot_report.json"
    started = time.perf_counter()
    report = {"status": "RUNNING", "run_id": run_id, "dataset": base.name,
              "purpose": "R3 computational pilot only", "population_ci_status": "not_established",
              "configuration": config, "levels": {}, "timing_seconds": {},
              "memory_measure": "process lifetime peak RSS; includes imported libraries and input shards"}
    write_report(report_path, report)
    print(f"R3 pilot output: {output}", flush=True)
    try:
        settings = config["v1_validation"].get("r3-pilot", {})
        if not isinstance(settings, dict):
            raise ValueError("v1_validation.r3-pilot must be a mapping")
        defaults = dict(block_length_windows=128, pilot_blocks_per_condition=8,
                        primary_blocks_per_condition=32, resampling_replicates=100,
                        tile_size=128)
        unknown = set(settings) - set(defaults)
        if unknown:
            raise ValueError(f"Unknown R3 pilot settings: {sorted(unknown)}")
        settings = {key: positive_int(settings.get(key, value), key) for key, value in defaults.items()}
        L, B, P, R, tile = (settings[k] for k in defaults)
        if B > P or B < 2 or R < 2 or tile > 512:
            raise ValueError("Require 2 <= pilot blocks <= primary blocks, replicates >=2, tile <=512")
        if B * L > 4096 or R > 1000:
            raise ValueError("This computational pilot is capped at 4096 rows/condition and 1000 replicates")
        seed = config["seed"]
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        report["settings"] = settings
        report["settings"]["selection_seed"] = seed
        report["settings"]["resampling_seed"] = seed + 1000
        report["settings"]["unrelated_main_r3_settings_used"] = False
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
        report["checkpoint"] = str(checkpoint_path)
        report["checkpoint_sha256"] = checkpoint_hash
        report["preflight_sha256"] = sha256_file(preflight_path)
        report["source_code_sha256"] = {
            "run_r3_pilot.py": sha256_file(Path(__file__)),
            "r3_pilot.py": sha256_file(Path(helper.__file__)),
        }
        report["configuration_sha256"] = hashlib.sha256(
            json.dumps(config, sort_keys=True).encode()).hexdigest()
        step = sampling_step_ns(config["preprocessing"]["downsample_freq"])
        raw_manifests, plans, selected, manifests = {}, {}, {}, {}
        for k, (condition, split) in enumerate(CONDITIONS.items()):
            print(f"Checking timestamps and freezing blocks: {condition} / {split}", flush=True)
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
            plans[condition], selected[condition] = make_block_plan(
                timestamps, step_ns=step, block_length=L, pilot_blocks=B,
                primary_blocks=P, seed=seed + k, window_size=int(raw["window_size"]),
            )
            raw_manifests[condition], manifests[condition] = raw, entry
            del timestamps
        report["block_plans"] = plans
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
            print(f"Computing pilot level {level}", flush=True)
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
            # arrays created by this unique pilot run after successful level output.
            for path in array_paths.values():
                path.unlink()
            del tables
            gc.collect()
        compute = sum(item["distance_seconds"] for item in report["levels"].values())
        report["primary_compute_projection"] = {
            "selected_rows_per_condition": P * L, "quadratic_scale_factor": (P / B) ** 2,
            "distance_seconds_estimate": compute * (P / B) ** 2,
            "scope": "rough compute-only scaling; excludes I/O and hardware/cache effects; not a runtime guarantee"}
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
        print(f"Pilot {report['status']}: {report_path}", flush=True)


if __name__ == "__main__":
    main()
