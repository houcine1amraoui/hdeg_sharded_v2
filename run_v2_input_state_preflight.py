"""Read-only input-state preflight. Writes reports only; never prepares model data."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.utils.get_folders_utils import get_dataset_path, get_processed_folder

POLICY = "cu_input_state_v1"
STATUSES = ["valid", "unknown", "unavailable", "missing", "invalid"]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def stamp(path):
    s = Path(path).stat()
    return {"bytes": s.st_size, "mtime_ns": s.st_mtime_ns}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def classify(text, kind, registry):
    # Explicit parsing: pandas' default NA vocabulary must not erase state tokens.
    tokens = text.str.strip()
    missing = tokens.isin(registry["missing_tokens"]).to_numpy()
    values = pd.to_numeric(tokens, errors="coerce").to_numpy(dtype=np.float64)
    unknown = (~missing) & (values == registry["state_codes"]["unknown"])
    unavailable = (~missing) & (values == registry["state_codes"]["unavailable"])
    candidate = ~(missing | unknown | unavailable)
    finite = np.isfinite(values)
    other_code = candidate & np.isin(
        values, registry["other_documented_state_codes"]
    )
    # A code collision on a continuous channel requires review; we do not assert
    # that a negative physical measurement is necessarily a categorical state.
    nonfinite = candidate & ~finite
    unexpected_binary = (
        candidate & finite & ~other_code & ~np.isin(values, [0.0, 1.0])
        if kind == "binary" else np.zeros(len(values), dtype=bool)
    )
    invalid = other_code | nonfinite | unexpected_binary
    valid = candidate & ~invalid
    masks = dict(zip(STATUSES, [valid, unknown, unavailable, missing, invalid]))
    reasons = {
        "other_documented_code_requires_review": other_code,
        "nonnumeric_or_nonfinite": nonfinite,
        "unexpected_binary_value": unexpected_binary,
    }
    return tokens, values, masks, reasons


def inspect(config, registry_path, chunk_size, report):
    prep = config["preprocessing"]
    require(prep["dataset_name"] == "CU" and not prep.get("merge_bre_cu", False),
            "Only single-home CU is supported")
    require(prep["downsample_freq"] == "1s", "Expected 1s input")
    require(prep.get("apply_filtering") is True, "Expected configured feature filtering")
    require(chunk_size > 0, "chunk_size must be positive")

    processed = Path(get_processed_folder(config))
    raw = Path(get_dataset_path(config))
    files = {
        "runner": Path(__file__), "config": Path("configs/config.yaml"),
        "registry": registry_path, "devices": processed / "devices.json",
        "timestamps": processed / "timestamps.npz",
        "selected_features": Path("configs/CU_selected_features.txt"),
    }
    hashes = {k: digest(p) for k, p in files.items()}
    report["source_hashes"] = hashes
    before = stamp(raw)
    report["raw_csv"] = {"path": str(raw), **before, "content_sha256": None}
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    require(registry["policy_version"] == POLICY, "Unsupported registry policy")
    require(registry["dataset_name"] == "CU", "Registry dataset mismatch")
    require(registry["status_order"] == STATUSES, "Registry status order mismatch")
    require(registry["state_codes"] == {"unknown": -9.7, "unavailable": -9.21},
            "Unexpected availability mappings")
    require(isinstance(registry["missing_tokens"], list) and
            all(isinstance(t, str) for t in registry["missing_tokens"]),
            "Missing-token vocabulary must be an explicit string list")
    other_codes = registry["other_documented_state_codes"]
    require(isinstance(other_codes, list) and
            all(isinstance(v, (int, float)) and np.isfinite(v) for v in other_codes),
            "Invalid documented state-code mapping")
    require(not set(other_codes) & {-9.7, -9.21}, "State-code categories overlap")
    channels = registry["channels"]
    names = [c["name"] for c in channels]
    require(len(names) == 23 and len(set(names)) == 23, "Expected 23 unique channels")
    require(all(c["type"] in ["binary", "continuous"] for c in channels),
            "Unsupported channel type")
    require(sum(c["type"] == "binary" for c in channels) == 18,
            "Expected 18 binary and five continuous channels")
    devices = json.loads(files["devices"].read_text(encoding="utf-8"))
    selected = [s.strip() for s in files["selected_features"].read_text().splitlines() if s.strip()]
    require(len(set(selected)) == len(selected), "Duplicate selected features")
    with raw.open(encoding="utf-8-sig", newline="") as f:
        headers = next(csv.reader(f))
    require(len(set(headers)) == len(headers), "Duplicate source headers")
    require(len(set(h.strip() for h in headers)) == len(headers),
            "Whitespace normalization would create header collisions")
    require(headers.count("Timestamp") == 1, "Expected one exact Timestamp header")
    exact_selected = [n for n in selected if n in headers and n != "Timestamp"]
    schema = {
        "raw_headers": headers,
        "header_sha256": hashlib.sha256(json.dumps(headers, ensure_ascii=False).encode()).hexdigest(),
        "registry_order": names, "processed_device_order": devices,
        "selected_exact_match_order": exact_selected,
        "unmatched_selected_names": [n for n in selected if n not in headers],
        "registry_matches_devices": names == devices,
        "registry_matches_selection_intersection": names == exact_selected,
        "missing_registry_headers": [n for n in names if n not in headers],
        "automatic_aliases_applied": False,
    }
    report["schema"] = schema
    require(names == devices, "Registry order differs from processed devices.json")
    require(names == exact_selected, "Registry differs from exact selection/source intersection")
    require(not schema["missing_registry_headers"], "Registry channel missing from source")

    with np.load(files["timestamps"], allow_pickle=False) as a:
        train = a["train"]
    require(train.ndim == 1 and train.dtype.kind == "M", "Expected 1D datetime64 train timestamps")
    require(len(train) > 0 and not np.isnat(train).any(), "Empty/invalid training timestamps")
    train_ns = train.astype("datetime64[ns]").astype(np.int64)
    require(np.all(np.diff(train_ns) == 10**9), "Training timestamps must be continuous at 1s")
    report["training"] = {"rows_expected": len(train), "rows_matched": 0,
                          "first": str(train[0]), "last": str(train[-1]),
                          "timezone": "Not inferred; naive dataset coordinates"}
    stats = {}
    for c in channels:
        stats[c["name"]] = {
            "type": c["type"], "counts": {s: 0 for s in STATUSES},
            "invalid_reasons": {}, "invalid_examples": [],
            "valid_minimum": None, "valid_maximum": None,
            "binary_counts": {"zero": 0, "one": 0} if c["type"] == "binary" else None,
        }
    report["channels"] = stats
    seen = np.zeros(len(train), dtype=bool)
    last = None
    scanned = 0
    for chunk_index, chunk in enumerate(pd.read_csv(
            raw, usecols=["Timestamp"] + names, dtype=str, keep_default_na=False,
            na_filter=False, chunksize=chunk_size, encoding="utf-8-sig"), 1):
        scanned += len(chunk)
        times = pd.to_datetime(chunk["Timestamp"], format="ISO8601", errors="raise")
        require(not times.isna().any() and times.dt.tz is None,
                "Missing or timezone-aware timestamp requires explicit resolution")
        ns = times.to_numpy(dtype="datetime64[ns]").astype(np.int64)
        positions = np.searchsorted(train_ns, ns)
        take = np.zeros(len(ns), dtype=bool)
        inside = positions < len(train_ns)
        take[inside] = train_ns[positions[inside]] == ns[inside]
        in_range = (ns >= train_ns[0]) & (ns <= train_ns[-1])
        require(not np.any(in_range & ~take), "Off-grid timestamp inside training interval")
        if not take.any():
            continue
        chosen, idx = ns[take], positions[take]
        require(np.all(np.diff(chosen) > 0) and (last is None or chosen[0] > last),
                "Training rows duplicated or out of order")
        require(not seen[idx].any(), "Repeated training timestamp")
        seen[idx] = True
        last = int(chosen[-1])
        for c in channels:
            name, kind = c["name"], c["type"]
            tokens, values, masks, reasons = classify(chunk.loc[take, name], kind, registry)
            s = stats[name]
            for key, mask in masks.items():
                s["counts"][key] += int(mask.sum())
            for key, mask in reasons.items():
                s["invalid_reasons"][key] = s["invalid_reasons"].get(key, 0) + int(mask.sum())
            for j in np.flatnonzero(masks["invalid"]):
                if len(s["invalid_examples"]) >= 5:
                    break
                s["invalid_examples"].append({"timestamp": str(np.datetime64(int(chosen[j]), "ns")),
                                              "token": str(tokens.iloc[j])[:160]})
            valid = values[masks["valid"]]
            if len(valid):
                lo, hi = float(valid.min()), float(valid.max())
                s["valid_minimum"] = lo if s["valid_minimum"] is None else min(lo, s["valid_minimum"])
                s["valid_maximum"] = hi if s["valid_maximum"] is None else max(hi, s["valid_maximum"])
                if kind == "binary":
                    s["binary_counts"]["zero"] += int((valid == 0).sum())
                    s["binary_counts"]["one"] += int((valid == 1).sum())
        report["training"]["rows_matched"] += len(chosen)
        if chunk_index % 10 == 0:
            print(f"Scanned {scanned:,} rows; matched {int(seen.sum()):,} training rows", flush=True)
    report["raw_csv"]["rows_scanned_for_timestamps"] = scanned
    require(seen.all(), f"Missing {int((~seen).sum())} expected training timestamps")
    require(stamp(raw) == before, "Source CSV size/mtime changed during scan")
    require(all(digest(p) == hashes[k] for k, p in files.items()),
            "A configuration/provenance input changed during the scan")
    blockers = []
    for name, s in stats.items():
        counts = s["counts"]
        require(sum(counts.values()) == len(train), f"Status accounting failure: {name}")
        s["fractions"] = {k: v / len(train) for k, v in counts.items()}
        eligible = counts["valid"] > 0 and counts["invalid"] == 0
        s["scaling_eligible"] = eligible
        s["constant_valid_training_values"] = (
            s["valid_minimum"] == s["valid_maximum"] if counts["valid"] else None)
        if not counts["valid"]:
            blockers.append(f"{name}: no eligible training values")
        if counts["invalid"]:
            blockers.append(f"{name}: {counts['invalid']} invalid/unresolved values")
        if not eligible:
            s["proposed_transform"] = None
        elif s["type"] == "binary":
            s["proposed_transform"] = {"method": "identity", "allowed_values": [0, 1]}
        else:
            low, high = s["valid_minimum"], s["valid_maximum"]
            s["proposed_transform"] = {"method": "training_valid_minmax", "offset": low,
                                      "denominator": high - low if high > low else 1.0,
                                      "constant_unit_denominator": high == low, "clip": False}
    report["blockers"] = blockers
    report["status"] = "BLOCKED" if blockers else "PASS"
    report["input_state_preflight_passed"] = not blockers
    report["scan_complete"] = True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project_root_dir", default=None)
    parser.add_argument("--registry", default=None)
    parser.add_argument("--chunk_size", type=int, default=50000)
    args = parser.parse_args()
    with open("configs/config.yaml") as f:
        config = yaml.safe_load(f)
    if args.project_root_dir:
        config["project_root_dir"] = args.project_root_dir
    registry = Path(args.registry or config["preprocessing"].get(
        "input_state_registry", "configs/CU_input_state_registry.json"))
    processed = Path(get_processed_folder(config))
    report = {
        "protocol": "v2_input_state_preflight_v1", "policy_version": POLICY,
        "status": "BLOCKED", "input_state_preflight_passed": False, "scan_complete": False,
        "configuration": config, "registry_path": str(registry), "chunk_size": args.chunk_size,
        "optimizer_steps": 0, "model_forward_calls": 0, "scientific_claim_decision": "NOT_EVALUATED",
        "limitations": [
            "Training values only; all CSV timestamps may be scanned. No held-out value diagnostics.",
            "Raw CSV identity uses size/mtime and header hash, not a full content hash.",
            "Valid means policy-eligible, not fresh or physically verified observations.",
            "Known categorical-code collisions on continuous channels require semantic review.",
            "Upstream filling and historical feature-selection information are not reconstructed.",
            "PASS concerns input eligibility only, not forecasting quality or scientific acceptance.",
            "No transformed arrays, scaler object, checkpoint, or existing output is modified.",
        ],
    }
    try:
        inspect(config, registry, args.chunk_size, report)
    except (ValueError, KeyError, TypeError, OSError, StopIteration) as exc:
        report["blockers"] = [f"{type(exc).__name__}: {exc}"]
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
    output = processed / "v2_validation" / "input_state_preflight" / run_id
    output.mkdir(parents=True, exist_ok=False)
    target = output / "input_state_preflight_report.json"
    with target.open("x") as f:
        json.dump(report, f, indent=2, allow_nan=False)
    print(f"Input-state preflight: {report['status']}\nReport: {target}")
    for message in report.get("blockers", []):
        print(f"  {message}")
    return 0 if report["input_state_preflight_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
