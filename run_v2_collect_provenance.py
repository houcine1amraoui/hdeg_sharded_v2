"""Collect small provenance evidence without changing research inputs."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import yaml

from src.utils.get_folders_utils import get_processed_folder

SPLITS = ("train", "val", "actor2_test", "actor1_test")
MIB = 1024 * 1024


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(MIB), b""):
            h.update(chunk)
    return h.hexdigest()


def serial(value):
    if isinstance(value, np.ndarray):
        return serial(value.tolist())
    if isinstance(value, np.generic):
        return serial(value.item())
    if isinstance(value, dict):
        return {str(k): serial(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serial(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def timestamp_summary(values, window, expected_ns):
    a = np.asarray(values)
    if a.ndim != 1 or a.dtype.kind not in "MUS":
        raise ValueError("Expected one-dimensional datetime64 or timestamp strings; numeric units are not guessed")
    a = a.astype("datetime64[ns]")
    if np.isnat(a).any():
        raise ValueError("NaT timestamps present")
    ns = a.astype(np.int64)
    delta = np.diff(ns)
    bad = delta != expected_ns
    # Pair i uses observations i through i+W inclusive (W transitions).
    prefix = np.r_[0, np.cumsum(bad, dtype=np.int64)]
    pairs = max(0, len(ns) - window)
    invalid = int(np.count_nonzero(prefix[window:] - prefix[:-window])) if pairs else 0
    unique = np.unique(ns)
    report = {
        "rows": len(ns), "first": str(a[0]) if len(a) else None,
        "last": str(a[-1]) if len(a) else None,
        "timezone": "not inferred from datetime64",
        "duplicate_timestamp_count": len(ns) - len(unique),
        "backward_step_count": int(np.count_nonzero(delta < 0)),
        "unexpected_step_count": int(bad.sum()),
        "unexpected_step_examples": [
            {"left_row": int(i), "left": str(a[i]), "right": str(a[i+1]), "step_ns": int(delta[i])}
            for i in np.flatnonzero(bad)[:20]
        ],
        "window_pair_count_by_row": pairs,
        "pairs_crossing_unexpected_steps": invalid,
        "regular_pairs": pairs - invalid,
        "decoded_ns_sha256": hashlib.sha256(ns.astype("<i8").tobytes()).hexdigest(),
    }
    return report, unique


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project_root_dir", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--record", action="append", default=[],
                        help="Existing JSON/TXT/LOG/YAML/MD execution record; repeat as needed")
    parser.add_argument("--history_file", default=None,
                        help="Completed provenance questionnaire JSON, if available")
    parser.add_argument("--hash_raw_csv", action="store_true",
                        help="Optionally stream-hash the configured raw CSV; never copy it")
    args = parser.parse_args()
    # Preserve the project's established configuration-loading convention.
    with open("configs/config.yaml", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if args.project_root_dir:
        config["project_root_dir"] = args.project_root_dir
    processed = Path(get_processed_folder(config)).resolve()
    project = Path(config["project_root_dir"]).resolve()
    prep = config["preprocessing"]
    if prep.get("merge_bre_cu", False):
        raise ValueError("This focused collector supports a single-home dataset only")
    window = int(prep["window_size"])
    if window <= 0:
        raise ValueError("window_size must be positive")
    # This collector is scoped to the audited one-second protocol.
    if prep["downsample_freq"] != "1s":
        raise ValueError("This focused collector expects downsample_freq: 1s")
    out = Path(args.output_dir).resolve() if args.output_dir else processed / "provenance_bundles"
    out.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
    destination = out / ("v2_provenance_" + run_id + ".zip")
    report = {
        "protocol": "v2_read_only_provenance_bundle_v1", "run_id": run_id,
        "status": "COLLECTED_WITH_UNRESOLVED_HISTORY",
        "scientific_claim_decision": "NOT_EVALUATED",
        "confirmation_eligibility": "NOT_ESTABLISHED_BY_COLLECTION",
        "optimizer_steps": 0, "model_forward_calls": 0,
        "processed_directory": str(processed), "resolved_configuration": config,
        "runtime": {"python": platform.python_version(), "numpy": np.__version__, "joblib": joblib.__version__},
        "files": [], "issues": [], "timestamp_summaries": {},
        "cross_split_timestamp_overlap": [],
        "limits": ["Current files do not prove historical execution", "No window tensors or checkpoints loaded",
                   "Scaler attributes cannot identify fitting rows", "Unknown histories remain unknown",
                   "Timestamp checks do not reconstruct raw preprocessing dependencies"],
    }
    questionnaire = {
        "status": "UNKNOWN_UNTIL_COMPLETED_BY_RESEARCHER",
        "feature_selection": {"rationale": None, "chosen_when": None,
                              "data_periods_inspected": None, "used_full_recording_statistics": None},
        "preprocessing": {"actual_command": None, "execution_date": None,
                          "source_version": None, "scaler_fit_rows_or_period": None,
                          "manual_changes_or_imputation": None},
        "training": {"actual_commands": None, "resume_sources": None,
                     "data_splits_used_for_updates": None, "checkpoint_selection_rule": None},
        "candidate_confirmation": {"period": None, "normal_operation_basis": None,
                                   "prior_fitting_use": None, "prior_diagnostic_or_selection_use": None,
                                   "reserved_for_other_levels": None},
        "instructions": "Fill only what you know; retain null for unknowns. This statement is self-reported, not verified execution evidence.",
    }
    total_small = 0
    with zipfile.ZipFile(destination, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as bundle:
        def put_json(name, value):
            bundle.writestr(name, json.dumps(serial(value), indent=2, allow_nan=False))

        def add(path, name, limit=2*MIB, timestamp=False):
            nonlocal total_small
            path = Path(path)
            row = {"source": str(path.resolve()), "archive_path": name}
            report["files"].append(row)
            if not path.is_file():
                row["status"] = "MISSING"
                return False
            size = path.stat().st_size
            row["bytes"] = size
            if size > limit or (not timestamp and total_small + size > 32*MIB):
                row["status"] = "SKIPPED_SIZE_LIMIT"
                return False
            before = digest(path)
            bundle.write(path, name)
            after = digest(path)
            row.update(status="COPIED" if before == after else "CHANGED_DURING_COLLECTION",
                       sha256=before, sha256_after=after)
            if not timestamp:
                total_small += size
            return before == after

        add("configs/config.yaml", "configuration/config.yaml")
        add(Path(__file__), "source/run_v2_collect_provenance.py")
        for relative in ["run_data_preparation.py", "src/data_preparation/data_preparation.py",
                         "src/data_preparation/prepare_windows_sharded.py", "src/utils/save_utils.py",
                         "src/utils/get_folders_utils.py", "src/training/train_hdeg_e2e.py",
                         "configs/" + prep["dataset_name"] + "_selected_features.txt"]:
            add(Path(relative), "current_source/" + relative)
        add(processed / "devices.json", "processed/devices.json")
        add(processed / "windows/manifest.json", "windows/manifest.json")
        for split in SPLITS:
            add(processed / "windows" / split / "manifest.json", "windows/" + split + "/manifest.json")

        timestamps = processed / "timestamps.npz"
        if add(timestamps, "processed/timestamps.npz", limit=128*MIB, timestamp=True):
            try:
                uniques = {}
                with np.load(timestamps, allow_pickle=False) as arrays:
                    for split in SPLITS:
                        try:
                            summary, unique = timestamp_summary(arrays[split], window, 10**9)
                            report["timestamp_summaries"][split] = summary
                            uniques[split] = unique
                            manifest_path = processed / "windows" / split / "manifest.json"
                            if manifest_path.is_file() and manifest_path.stat().st_size <= 2*MIB:
                                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                                summary["manifest_comparison"] = {
                                    "num_timesteps_matches": manifest.get("num_timesteps") == summary["rows"],
                                    "num_samples_matches": manifest.get("num_samples") == summary["window_pair_count_by_row"],
                                    "window_size_matches": manifest.get("window_size") == window,
                                    "scope": "Counts only; does not verify shard tensor contents",
                                }
                        except Exception as exc:
                            report["issues"].append({"component": "timestamps/" + split, "error": str(exc)})
                for i, left in enumerate(SPLITS):
                    for right in SPLITS[i+1:]:
                        if left in uniques and right in uniques:
                            overlap = np.intersect1d(uniques[left], uniques[right], assume_unique=True)
                            report["cross_split_timestamp_overlap"].append({
                                "left": left, "right": right, "shared_observation_timestamps": len(overlap),
                                "interpretation": "Processed timestamps only; not proof of statistical independence"})
                if digest(timestamps) != next(r["sha256"] for r in report["files"] if r["archive_path"] == "processed/timestamps.npz"):
                    report["issues"].append({"component": "timestamps", "error": "Changed while being summarized"})
            except Exception as exc:
                report["issues"].append({"component": "timestamps", "error": str(exc)})

        scaler_path = processed / "scaler.pkl"
        scaler_info = {"path": str(scaler_path), "status": "MISSING", "fit_history_verified": False}
        if scaler_path.is_file():
            try:
                if scaler_path.stat().st_size > 16*MIB:
                    raise ValueError("Scaler exceeds 16 MiB metadata-collection limit")
                before = digest(scaler_path)
                # Only the researcher's existing, trusted locally generated scaler is loaded.
                scaler = joblib.load(scaler_path)
                scaler_info.update(status="ATTRIBUTES_READ", sha256=before,
                                   object_type=type(scaler).__module__ + "." + type(scaler).__name__)
                for attr in ["n_features_in_", "n_samples_seen_", "feature_names_in_", "feature_range",
                             "data_min_", "data_max_", "data_range_", "scale_", "min_", "clip"]:
                    if hasattr(scaler, attr):
                        scaler_info[attr] = serial(getattr(scaler, attr))
                scaler_info["sha256_after"] = digest(scaler_path)
                if before != scaler_info["sha256_after"]:
                    scaler_info["status"] = "CHANGED_DURING_COLLECTION"
            except Exception as exc:
                scaler_info.update(status="UNREADABLE", error=str(exc))
        put_json("scaler_metadata.json", scaler_info)

        # Narrow allowlist: no recursive collection of data, checkpoints, or arbitrary home files.
        records = [processed / "v1_validation/preflight.json"]
        for pattern in ["hdeg_checkpoints/runs/*/run_configuration.json", "hdeg_checkpoints/runs/*/completed.json",
                        "experiments/*/experiment_manifest.json", "experiments/*/configuration_snapshot.json",
                        "v2_validation/preflight/*/preflight_report.json"]:
            records.extend(sorted(processed.glob(pattern)))
        records.extend(Path(x) for x in args.record)
        records = list(dict.fromkeys(records))
        if len(records) > 100:
            report["issues"].append({"component": "records", "error": "Only first 100 candidate records collected", "candidates": len(records)})
        for i, path in enumerate(records[:100]):
            if path.suffix.lower() not in {".json", ".txt", ".log", ".yaml", ".yml", ".md"}:
                report["issues"].append({"component": "record", "path": str(path), "error": "Extension not allowed"})
                continue
            add(path, "records/" + str(i).zfill(3) + "_" + path.name)
        if args.history_file:
            history = Path(args.history_file)
            if history.suffix.lower() != ".json":
                report["history_status"] = "NOT_COLLECTED_EXPECTED_JSON"
            else:
                copied = add(history, "researcher_history.json")
                report["history_status"] = "SUPPLIED_FILE_NOT_INDEPENDENTLY_VERIFIED" if copied else "NOT_COLLECTED"
        else:
            report["history_status"] = "UNKNOWN_QUESTIONNAIRE_INCLUDED"
        put_json("provenance_questionnaire.json", questionnaire)

        raw = project / prep["dataset_folder"] / (prep["dataset_name"] + "Master.csv")
        raw_info = {"path": str(raw), "present": raw.is_file(), "copied": False}
        if raw.is_file():
            raw_info["bytes"] = raw.stat().st_size
            if args.hash_raw_csv:
                raw_info["sha256"] = digest(raw)
            else:
                raw_info["sha256_status"] = "NOT_REQUESTED"
        report["raw_source"] = raw_info
        report["missing_or_skipped_files"] = sum(r["status"] != "COPIED" for r in report["files"])
        put_json("collection_report.json", report)
    print("Provenance collection finished; scientific eligibility is NOT assessed.")
    print("Missing/skipped/changed files:", report["missing_or_skipped_files"])
    print("Collection issues:", len(report["issues"]))
    print("Bundle:", destination)
    print("Bundle SHA256:", digest(destination))
    print("Complete provenance_questionnaire.json separately if history is known; do not invent missing records.")


if __name__ == "__main__":
    main()
