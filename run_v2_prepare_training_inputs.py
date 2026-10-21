"""Prepare training-only value/status rows under the verified CU input policy."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

import run_v2_input_state_preflight as preflight
from src.utils.get_folders_utils import get_dataset_path, get_processed_folder


def require(condition, message):
    if not condition:
        raise ValueError(message)


def write_json(path, data):
    """Publish a new file; all callers use a newly created run directory."""
    require(not path.exists(), f"Refusing to overwrite {path}")
    temp = path.with_suffix(path.suffix + ".partial")
    with temp.open("x", encoding="utf-8") as f:
        json.dump(data, f, indent=2, allow_nan=False)
    os.replace(temp, path)


def verify_inputs(config, registry_path, report_path):
    processed = Path(get_processed_folder(config))
    raw = Path(get_dataset_path(config))
    files = {
        "preparation_runner": Path(__file__),
        "preflight_report": report_path,
        "runner": Path(preflight.__file__),
        "config": Path("configs/config.yaml"),
        "registry": registry_path,
        "devices": processed / "devices.json",
        "timestamps": processed / "timestamps.npz",
        "selected_features": Path("configs/CU_selected_features.txt"),
    }
    hashes = {key: preflight.digest(path) for key, path in files.items()}
    report = json.loads(report_path.read_text(encoding="utf-8"))
    require(report["protocol"] == "v2_input_state_preflight_v1", "Unsupported preflight protocol")
    require(report["policy_version"] == preflight.POLICY == "cu_input_state_v1", "Policy mismatch")
    require(report["status"] == "PASS" and report["input_state_preflight_passed"] is True
            and report["scan_complete"] is True and report["blockers"] == [],
            "A complete, successful input-state preflight is required")
    require(config == report["configuration"], "Effective configuration differs from preflight")
    for key, expected in report["source_hashes"].items():
        require(key in hashes and hashes[key] == expected, f"Preflight dependency hash differs: {key}")
    require(set(report["source_hashes"]) == {
        "runner", "config", "registry", "devices", "timestamps", "selected_features"
    }, "Incomplete preflight dependency hashes")
    require(config["preprocessing"]["dataset_name"] == "CU"
            and not config["preprocessing"].get("merge_bre_cu", False)
            and config["preprocessing"]["downsample_freq"] == "1s",
            "Only single-home CU at 1s is supported")
    before = preflight.stamp(raw)
    require(str(raw) == report["raw_csv"]["path"], "Source path differs from preflight")
    require(before == {k: report["raw_csv"][k] for k in ["bytes", "mtime_ns"]},
            "Source size/mtime differs from preflight; rerun preflight")
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    require(registry["policy_version"] == preflight.POLICY, "Registry policy mismatch")
    require(registry["status_order"] == preflight.STATUSES, "Status vocabulary/order mismatch")
    names = [channel["name"] for channel in registry["channels"]]
    require(len(names) == len(set(names)) == 23, "Expected 23 unique channels")
    require(names == json.loads(files["devices"].read_text()), "Device order mismatch")
    with raw.open(encoding="utf-8-sig", newline="") as f:
        headers = next(csv.reader(f))
    require(headers == report["schema"]["raw_headers"], "Raw header differs from preflight")
    selected = [s.strip() for s in files["selected_features"].read_text().splitlines() if s.strip()]
    require(names == [s for s in selected if s in headers and s != "Timestamp"],
            "Selected source intersection differs from registry")
    require(names == report["schema"]["registry_order"]
            == report["schema"]["processed_device_order"]
            == report["schema"]["selected_exact_match_order"], "Preflight channel order mismatch")
    require(set(names) == set(report["channels"]), "Preflight channel set differs")
    header_hash = hashlib.sha256(json.dumps(headers, ensure_ascii=False).encode()).hexdigest()
    require(header_hash == report["schema"]["header_sha256"], "Header digest mismatch")
    with np.load(files["timestamps"], allow_pickle=False) as a:
        train = a["train"]
    require(train.ndim == 1 and train.dtype.kind == "M" and len(train) > 0
            and not np.isnat(train).any(), "Invalid saved training timestamp array")
    train = train.astype("datetime64[ns]")
    train_ns = train.astype(np.int64)
    require(np.all(np.diff(train_ns) == 10**9), "Training timestamps must be continuous at 1s")
    require(len(train) == report["training"]["rows_expected"] == report["training"]["rows_matched"]
            and str(train[0]) == report["training"]["first"]
            and str(train[-1]) == report["training"]["last"], "Preflight training period mismatch")
    transforms = {}
    for channel in registry["channels"]:
        name, kind = channel["name"], channel["type"]
        s = report["channels"][name]
        require(kind == s["type"] and s["scaling_eligible"] is True, f"Ineligible channel: {name}")
        require(set(s["counts"]) == set(preflight.STATUSES)
                and all(isinstance(v, int) and v >= 0 for v in s["counts"].values())
                and sum(s["counts"].values()) == len(train)
                and s["counts"]["valid"] > 0 and s["counts"]["invalid"] == 0,
                f"Invalid preflight counts: {name}")
        low, high = s["valid_minimum"], s["valid_maximum"]
        require(np.isfinite(low) and np.isfinite(high) and low <= high, f"Invalid range: {name}")
        if kind == "binary":
            expected = {"method": "identity", "allowed_values": [0, 1]}
            require(low in [0, 1] and high in [0, 1], f"Invalid binary range: {name}")
        else:
            require(kind == "continuous", f"Unsupported type: {kind}")
            expected = {"method": "training_valid_minmax", "offset": low,
                        "denominator": high - low if high > low else 1.0,
                        "constant_unit_denominator": high == low, "clip": False}
            require(np.isfinite(expected["denominator"]) and expected["denominator"] > 0,
                    f"Invalid scaling denominator: {name}")
        require(expected == s["proposed_transform"], f"Preflight transform inconsistent: {name}")
        transforms[name] = expected
    return processed, raw, before, files, hashes, report, registry, names, train, transforms


class ShardWriter:
    """Fixed-size row shards independent of CSV chunk boundaries."""
    def __init__(self, directory, rows, channels):
        self.directory = directory
        self.capacity = rows
        self.values = np.empty((rows, channels), dtype=np.float32)
        self.status = np.empty((rows, channels), dtype=np.uint8)
        self.times = np.empty(rows, dtype="datetime64[ns]")
        self.used = 0
        self.total = 0
        self.entries = []

    def append(self, values, status, times):
        offset = 0
        while offset < len(times):
            count = min(self.capacity - self.used, len(times) - offset)
            target = slice(self.used, self.used + count)
            source = slice(offset, offset + count)
            self.values[target], self.status[target], self.times[target] = (
                values[source], status[source], times[source])
            self.used += count
            offset += count
            if self.used == self.capacity:
                self.flush()

    def flush(self):
        if not self.used:
            return
        name = f"shard_{len(self.entries):06d}.npz"
        path = self.directory / name
        temp = self.directory / (name + ".partial")
        values, status, times = self.values[:self.used], self.status[:self.used], self.times[:self.used]
        with temp.open("xb") as f:
            np.savez_compressed(f, values=values, status=status, timestamps=times)
        # Check serialization before accepting each shard.
        with np.load(temp, allow_pickle=False) as saved:
            for key, expected in [("values", values), ("status", status), ("timestamps", times)]:
                require(saved[key].dtype == expected.dtype and np.array_equal(saved[key], expected),
                        f"Shard round-trip verification failed: {name}/{key}")
        require(not path.exists(), f"Refusing to overwrite {path}")
        os.replace(temp, path)
        self.entries.append({"file": name, "index": len(self.entries), "rows": self.used,
                             "start": self.total, "stop": self.total + self.used,
                             "first_timestamp": str(times[0]), "last_timestamp": str(times[-1]),
                             "sha256": preflight.digest(path), "bytes": path.stat().st_size})
        self.total += self.used
        self.used = 0
        print(f"Saved {name}: {self.total:,} training rows", flush=True)


def prepare(context, output, chunk_size, shard_size):
    processed, raw, before, files, hashes, report, registry, names, train, transforms = context
    train_ns = train.astype(np.int64)
    writer = ShardWriter(output, shard_size, len(names))
    counts = {name: {s: 0 for s in preflight.STATUSES} for name in names}
    extrema = {name: [None, None] for name in names}
    binary_counts = {name: {"zero": 0, "one": 0} for name in names
                     if report["channels"][name]["type"] == "binary"}
    cursor = scanned = 0
    for chunk in pd.read_csv(raw, usecols=["Timestamp"] + names, dtype=str,
                             keep_default_na=False, na_filter=False, chunksize=chunk_size,
                             encoding="utf-8-sig"):
        scanned += len(chunk)
        times = pd.to_datetime(chunk["Timestamp"], format="ISO8601", errors="raise")
        require(not times.isna().any() and times.dt.tz is None,
                "Invalid/timezone-aware CSV timestamps")
        ns = times.to_numpy(dtype="datetime64[ns]").astype(np.int64)
        take = (ns >= train_ns[0]) & (ns <= train_ns[-1])
        if not take.any():
            continue
        chosen = ns[take]
        require(np.array_equal(chosen, train_ns[cursor:cursor + len(chosen)]),
                "Training timestamps differ: gap, duplicate, order error, or off-grid row")
        values_out = np.zeros((len(chosen), len(names)), dtype=np.float32)
        status_out = np.empty(values_out.shape, dtype=np.uint8)
        for j, channel in enumerate(registry["channels"]):
            name, kind = channel["name"], channel["type"]
            tokens, values, masks, reasons = preflight.classify(chunk.loc[take, name], kind, registry)
            require(not masks["invalid"].any(), f"Unexpected invalid value in {name}")
            require(np.all(sum(mask.astype(np.uint8) for mask in masks.values()) == 1),
                    f"Nonexclusive status classification: {name}")
            for code, state in enumerate(preflight.STATUSES):
                mask = masks[state]
                counts[name][state] += int(mask.sum())
                status_out[mask, j] = code
            valid = masks["valid"]
            raw_valid = values[valid]
            if len(raw_valid):
                lo, hi = float(raw_valid.min()), float(raw_valid.max())
                a, b = extrema[name]
                extrema[name] = [lo if a is None else min(a, lo), hi if b is None else max(b, hi)]
                require(lo >= report["channels"][name]["valid_minimum"]
                        and hi <= report["channels"][name]["valid_maximum"],
                        f"Training range differs from preflight: {name}")
                if kind == "binary":
                    scaled = raw_valid
                    binary_counts[name]["zero"] += int((raw_valid == 0).sum())
                    binary_counts[name]["one"] += int((raw_valid == 1).sum())
                else:
                    t = transforms[name]
                    scaled = (raw_valid - t["offset"]) / t["denominator"]
                require(np.isfinite(scaled).all(), f"Nonfinite scaled values: {name}")
                values_out[valid, j] = scaled.astype(np.float32)
        require(np.isfinite(values_out).all(), "Nonfinite float32 output")
        require(np.all(values_out[status_out != 0] == 0), "Nonzero invalid-state placeholder")
        writer.append(values_out, status_out, chosen.astype("datetime64[ns]"))
        cursor += len(chosen)
    require(cursor == len(train), "Not all training timestamps were found")
    require(scanned == report["raw_csv"]["rows_scanned_for_timestamps"],
            "Source row count differs from preflight")
    for name in names:
        previous = report["channels"][name]
        require(counts[name] == previous["counts"], f"Status counts differ from preflight: {name}")
        require(extrema[name] == [previous["valid_minimum"], previous["valid_maximum"]],
                f"Eligible range differs from preflight: {name}")
        if name in binary_counts:
            require(binary_counts[name] == previous["binary_counts"],
                    f"Binary counts differ from preflight: {name}")
    require(preflight.stamp(raw) == before, "CSV changed during preparation")
    require(all(preflight.digest(p) == hashes[k] for k, p in files.items()),
            "A dependency changed during preparation")
    writer.flush()
    require(writer.total == len(train), "Shard row count mismatch")
    require(all(preflight.digest(output / s["file"]) == s["sha256"] for s in writer.entries),
            "A written shard changed before completion")
    provenance = {"source_hashes": hashes, "preflight_report_path": str(files["preflight_report"]),
                  "raw_csv": report["raw_csv"], "header_sha256": report["schema"]["header_sha256"],
                  "raw_identity_limit": "No raw content hash; size/mtime, headers, timestamps, and training summaries checked"}
    metadata = {"policy_version": preflight.POLICY, "split": "train", "devices": names,
                "status_codes": {s: i for i, s in enumerate(preflight.STATUSES)},
                "nonvalid_placeholder": 0.0, "transforms": transforms,
                "training": report["training"], "registry": registry, **provenance}
    write_json(output / "transformation_metadata.json", metadata)
    execution = {"status": "COMPLETE", "policy_version": preflight.POLICY, "split": "train",
                 "rows_written": writer.total, "shards_written": len(writer.entries),
                 "chunk_size": chunk_size, "shard_size": shard_size,
                 "counts": counts, "eligible_raw_ranges": extrema, "binary_counts": binary_counts,
                 "configuration": report["configuration"], "optimizer_steps": 0, "model_forward_calls": 0,
                 "scientific_claim_decision": "NOT_EVALUATED", **provenance}
    write_json(output / "preparation_report.json", execution)
    # The manifest is the completion marker and is published LAST.
    manifest = {"format": "hdeg_input_state_rows_v1", "status": "COMPLETE", "split": "train",
                "policy_version": preflight.POLICY, "num_rows": writer.total, "num_channels": len(names),
                "devices": names, "status_codes": metadata["status_codes"],
                "arrays": {"values": "float32 [rows,23]", "status": "uint8 [rows,23]",
                           "timestamps": "datetime64[ns] [rows]"},
                "shard_size": shard_size, "shards": writer.entries,
                "metadata": {name: {"file": name, "sha256": preflight.digest(output / name)}
                             for name in ["transformation_metadata.json", "preparation_report.json"]},
                "preflight_report_sha256": hashes["preflight_report"],
                "registry_sha256": hashes["registry"], "windows_constructed": False}
    write_json(output / "manifest.json", manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project_root_dir", default=None)
    parser.add_argument("--preflight_report", required=True)
    parser.add_argument("--registry", default=None)
    parser.add_argument("--chunk_size", type=int, default=None)
    parser.add_argument("--shard_size", type=int, default=None)
    args = parser.parse_args()
    output = None
    try:
        with open("configs/config.yaml") as f:
            config = yaml.safe_load(f)
        if args.project_root_dir:
            config["project_root_dir"] = args.project_root_dir
        settings = config.get("input_state_preparation", {})
        chunk_size = args.chunk_size if args.chunk_size is not None else settings.get("chunk_size", 50000)
        shard_size = args.shard_size if args.shard_size is not None else settings.get("shard_size", 65536)
        require(type(chunk_size) is int and chunk_size > 0, "chunk_size must be a positive integer")
        require(type(shard_size) is int and shard_size > 0, "shard_size must be a positive integer")
        registry_path = Path(args.registry or config["preprocessing"].get(
            "input_state_registry", "configs/CU_input_state_registry.json"))
        context = verify_inputs(config, registry_path, Path(args.preflight_report))
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
        output = context[0] / "input_states" / preflight.POLICY / "train" / run_id
        output.mkdir(parents=True, exist_ok=False)
        print(f"Training-only input preparation\nOutput: {output}\nRows: {len(context[8]):,}", flush=True)
        prepare(context, output, chunk_size, shard_size)
        print(f"COMPLETE\nManifest: {output / 'manifest.json'}")
        return 0
    except (Exception, KeyboardInterrupt) as exc:
        message = f"{type(exc).__name__}: {exc}"
        print(f"Preparation failed: {message}", file=sys.stderr)
        if output is not None:
            # Never publish a completion marker for a failed/interrupted run.
            if not (output / "manifest.json").exists():
                try:
                    write_json(output / "failure_report.json", {"status": "FAILED", "error": message})
                except OSError:
                    pass
                print(f"Incomplete output: {output}\nDo not consume it without a COMPLETE manifest.", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
