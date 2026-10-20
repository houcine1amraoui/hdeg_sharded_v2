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

from src.utils.get_folders_utils import (
    get_dataset_path,
    get_processed_folder,
)


# Meanings established by the supplied CUStatesMap.json.
STATE_CODES = {
    "unknown": -9.7,
    "unavailable": -9.21,
}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def stamp(path):
    s = Path(path).stat()
    return {"bytes": s.st_size, "mtime_ns": s.st_mtime_ns}


def time_text(ns):
    return str(np.datetime64(int(ns), "ns"))


class Runs:
    """Runs of matching observations, broken by nonmatches or timestamp gaps."""

    def __init__(self):
        self.run_count = 0
        self.longest = 0
        self.longest_start = None
        self.longest_end = None
        self.tail_length = 0
        self.tail_start = None
        self.last_timestamp = None

    def update(self, mask, times):
        if len(mask) == 0:
            return

        # Handle separate continuous time segments independently.
        cuts = np.r_[0, np.flatnonzero(np.diff(times) != 10**9) + 1, len(times)]

        for left, right in zip(cuts[:-1], cuts[1:]):
            m = mask[left:right]
            t = times[left:right]
            continuity = (
                self.last_timestamp is not None
                and int(t[0]) - self.last_timestamp == 10**9
            )

            edges = np.diff(np.r_[False, m, False].astype(np.int8))
            starts = np.flatnonzero(edges == 1)
            ends = np.flatnonzero(edges == -1)  # Exclusive.
            self.run_count += len(starts)

            new_tail_length = 0
            new_tail_start = None

            for start, end in zip(starts, ends):
                length = int(end - start)
                start_time = int(t[start])

                if start == 0 and continuity and self.tail_length:
                    length += self.tail_length
                    start_time = self.tail_start
                    self.run_count -= 1

                end_time = int(t[end - 1])

                if length > self.longest:
                    self.longest = length
                    self.longest_start = start_time
                    self.longest_end = end_time

                if end == len(m):
                    new_tail_length = length
                    new_tail_start = start_time

            self.tail_length = new_tail_length
            self.tail_start = new_tail_start
            self.last_timestamp = int(t[-1])

    def report(self):
        return {
            "run_count": self.run_count,
            "longest_run_observations": self.longest,
            "longest_start": (
                time_text(self.longest_start)
                if self.longest_start is not None else None
            ),
            "longest_end": (
                time_text(self.longest_end)
                if self.longest_end is not None else None
            ),
            "longest_elapsed_seconds": (
                (self.longest_end - self.longest_start) / 1e9
                if self.longest_start is not None else None
            ),
        }


class Channel:
    def __init__(self):
        self.rows = 0
        self.missing = 0
        self.nonnumeric = 0
        self.infinite = 0
        self.counts = {k: 0 for k in STATE_CODES}
        self.runs = {
            k: Runs() for k in [*STATE_CODES, "unknown_or_unavailable"]
        }
        self.remaining_count = 0
        self.mean = 0.0
        self.m2 = 0.0
        self.minimum = None
        self.maximum = None
        self.adjacent_remaining_pairs = 0
        self.remaining_transitions = 0
        self.last_time = None
        self.last_value = None
        self.last_remaining = False
        self.daily = {}

    def update(self, text, times, days):
        # Reproduce pandas' default missing-token interpretation from read_csv.
        missing = text.isna().to_numpy()
        values = pd.to_numeric(text, errors="coerce").to_numpy(dtype=np.float64)

        finite = np.isfinite(values)
        masks = {k: values == code for k, code in STATE_CODES.items()}
        special = masks["unknown"] | masks["unavailable"]
        remaining = finite & ~special

        self.rows += len(values)
        self.missing += int(missing.sum())
        self.nonnumeric += int((np.isnan(values) & ~missing).sum())
        self.infinite += int(np.isinf(values).sum())

        for key, mask in masks.items():
            self.counts[key] += int(mask.sum())
            self.runs[key].update(mask, times)
        self.runs["unknown_or_unavailable"].update(special, times)

        # Stable combined population moments; no model fitting occurs.
        selected = values[remaining]
        if len(selected):
            n = len(selected)
            mean = float(selected.mean())
            m2 = float(np.sum((selected - mean) ** 2))
            old_n = self.remaining_count
            total = old_n + n
            delta = mean - self.mean
            self.m2 += m2 + delta**2 * old_n * n / total
            self.mean += delta * n / total
            self.remaining_count = total

            low, high = float(selected.min()), float(selected.max())
            self.minimum = low if self.minimum is None else min(self.minimum, low)
            self.maximum = high if self.maximum is None else max(self.maximum, high)

        # Only directly adjacent, one-second, remaining-value pairs count.
        eligible = (
            remaining[1:] & remaining[:-1]
            & (np.diff(times) == 10**9)
        )
        self.adjacent_remaining_pairs += int(eligible.sum())
        self.remaining_transitions += int(
            (eligible & (values[1:] != values[:-1])).sum()
        )

        if (
            self.last_time is not None
            and int(times[0]) - self.last_time == 10**9
            and self.last_remaining
            and remaining[0]
        ):
            self.adjacent_remaining_pairs += 1
            self.remaining_transitions += int(values[0] != self.last_value)

        self.last_time = int(times[-1])
        self.last_value = float(values[-1]) if finite[-1] else None
        self.last_remaining = bool(remaining[-1])

        for day in np.unique(days):
            take = days == day
            key = str(day)
            row = self.daily.setdefault(
                key, {"rows": 0, "unknown": 0, "unavailable": 0}
            )
            row["rows"] += int(take.sum())
            for state, mask in masks.items():
                row[state] += int((take & mask).sum())

    def report(self):
        return {
            "training_rows": self.rows,
            "missing_token_count": self.missing,
            "nonnumeric_nonmissing_count": self.nonnumeric,
            "infinite_count": self.infinite,
            "state_counts": self.counts,
            "state_fractions_of_training_rows": {
                k: v / self.rows if self.rows else None
                for k, v in self.counts.items()
            },
            "runs": {k: v.report() for k, v in self.runs.items()},
            "remaining_finite_values": {
                "count": self.remaining_count,
                "minimum": self.minimum,
                "maximum": self.maximum,
                "mean": self.mean if self.remaining_count else None,
                "population_variance": (
                    self.m2 / self.remaining_count
                    if self.remaining_count else None
                ),
                "adjacent_one_second_pairs": self.adjacent_remaining_pairs,
                "exact_value_transitions": self.remaining_transitions,
                "transition_fraction": (
                    self.remaining_transitions / self.adjacent_remaining_pairs
                    if self.adjacent_remaining_pairs else None
                ),
                "interpretation": (
                    "Excludes only unknown/unavailable and nonfinite values. "
                    "Other encoded categorical states may remain; these are "
                    "not automatically valid physical measurements."
                ),
            },
            "daily_state_counts": self.daily,
        }


def main():
    parser = argparse.ArgumentParser(
        description="Read-only CU training-period raw-input audit."
    )
    parser.add_argument("--project_root_dir", default=None)
    parser.add_argument("--chunk_size", type=int, default=50000)
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()

    with open("configs/config.yaml", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if args.project_root_dir:
        config["project_root_dir"] = args.project_root_dir

    prep = config["preprocessing"]
    if (
        prep["dataset_name"] != "CU"
        or prep.get("merge_bre_cu", False)
        or prep["downsample_freq"] != "1s"
    ):
        raise ValueError("This focused audit expects single-home CU at 1s.")
    if args.chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")

    processed = Path(get_processed_folder(config))
    raw = Path(get_dataset_path(config))
    timestamps_path = processed / "timestamps.npz"
    devices_path = processed / "devices.json"
    features_path = Path("configs/CU_selected_features.txt")

    before = stamp(raw)

    with np.load(timestamps_path, allow_pickle=False) as archive:
        a = archive["train"]
        if a.dtype.kind != "M":
            raise ValueError("Expected datetime64 training timestamps.")
        if np.isnat(a).any():
            raise ValueError("Training timestamps contain NaT.")
        train_ns = a.astype("datetime64[ns]").astype(np.int64)

    if len(train_ns) == 0 or np.any(np.diff(train_ns) != 10**9):
        raise ValueError("Expected nonempty, continuous 1s training timestamps.")

    selected = [
        line.strip()
        for line in features_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    devices = json.loads(devices_path.read_text(encoding="utf-8"))

    with raw.open(newline="", encoding="utf-8-sig") as f:
        headers = next(csv.reader(f))

    if len(set(headers)) != len(headers):
        raise ValueError("Duplicate raw CSV headers require explicit resolution.")
    if headers.count("Timestamp") != 1:
        raise ValueError("Expected exactly one Timestamp column.")

    matching = []
    audited_headers = set()

    for name in dict.fromkeys(selected + devices):
        exact = [h for h in headers if h == name]
        whitespace = [h for h in headers if h.strip() == name]
        # Suggestion only: do not repair a feature name or alter preprocessing.
        quote_candidate = [
            h for h in headers if h.strip() == name.rstrip('"')
        ] if name.endswith('"') else []

        candidates = list(dict.fromkeys(exact + whitespace + quote_candidate))
        audited_headers.update(candidates)
        matching.append({
            "requested_name": name,
            "in_selected_feature_list": name in selected,
            "in_processed_device_list": name in devices,
            "exact_headers": exact,
            "whitespace_normalized_candidates": whitespace,
            "trailing_quote_candidates": quote_candidate,
            "automatic_correction_applied": False,
        })

    columns = [h for h in headers if h in audited_headers]
    if not columns:
        raise ValueError("No requested or candidate feature headers matched.")

    stats = {name: Channel() for name in columns}
    seen = np.zeros(len(train_ns), dtype=bool)
    last_training_time = None
    scanned_rows = 0
    training_rows = 0

    reader = pd.read_csv(
        raw,
        usecols=["Timestamp"] + columns,
        dtype=str,
        chunksize=args.chunk_size,
        encoding="utf-8-sig",
    )

    for chunk_number, chunk in enumerate(reader, 1):
        scanned_rows += len(chunk)
        times = pd.to_datetime(chunk["Timestamp"], errors="raise")
        if times.isna().any():
            raise ValueError("CSV contains missing timestamps.")
        if times.dt.tz is not None:
            raise ValueError("Timezone-aware CSV requires explicit alignment.")

        ns = times.to_numpy(dtype="datetime64[ns]").astype(np.int64)
        positions = np.searchsorted(train_ns, ns)
        inside = positions < len(train_ns)
        take = np.zeros(len(ns), dtype=bool)
        take[inside] = train_ns[positions[inside]] == ns[inside]

        if not take.any():
            continue

        chosen = ns[take]
        positions = positions[take]

        if (
            np.any(np.diff(chosen) <= 0)
            or (
                last_training_time is not None
                and chosen[0] <= last_training_time
            )
        ):
            raise ValueError("Training CSV rows are duplicated or out of order.")
        if seen[positions].any():
            raise ValueError("Repeated training timestamp in CSV.")
        seen[positions] = True
        last_training_time = int(chosen[-1])

        # Convert sensor values only after selecting training rows.
        training_chunk = chunk.loc[take]
        days = chosen.astype("datetime64[ns]").astype("datetime64[D]")
        for name in columns:
            stats[name].update(training_chunk[name], chosen, days)

        training_rows += len(chosen)
        if chunk_number % 10 == 0:
            print(
                f"Scanned {scanned_rows:,} CSV rows; "
                f"audited {training_rows:,} training rows.",
                flush=True,
            )

    if not seen.all():
        raise ValueError(
            f"CSV is missing {int((~seen).sum()):,} saved training timestamps."
        )
    if stamp(raw) != before:
        raise RuntimeError("CSV size/mtime changed during the audit.")

    report = {
        "protocol": "v2_training_raw_input_audit_v1",
        "status": "COMPLETE",
        "scientific_claim_decision": "NOT_EVALUATED",
        "optimizer_steps": 0,
        "model_forward_calls": 0,
        "scope": "Header inspection and training-period value statistics only",
        "state_code_source": "Supplied Misc/CUStatesMap.json",
        "state_codes": STATE_CODES,
        "configuration": config,
        "raw_csv": {
            "path": str(raw),
            **before,
            "rows_scanned_for_timestamp_matching": scanned_rows,
            "content_sha256": None,
            "identity_limit": "Size/mtime checked; full CSV hash not computed",
        },
        "training": {
            "rows_expected": len(train_ns),
            "rows_matched": training_rows,
            "first": time_text(train_ns[0]),
            "last": time_text(train_ns[-1]),
            "timezone": "No timezone inferred",
        },
        "source_hashes": {
            "runner": sha256(__file__),
            "config": sha256("configs/config.yaml"),
            "timestamps": sha256(timestamps_path),
            "devices": sha256(devices_path),
            "selected_features": sha256(features_path),
        },
        "feature_matching": matching,
        "channels": {name: value.report() for name, value in stats.items()},
        "limitations": [
            "CUMaster is an upstream processed/filled dataset.",
            "No values replaced, imputed, normalized, or saved as training data.",
            "No validation/test value statistics computed.",
            "All-missing training columns need not be all-missing globally.",
            "Counts do not prove the cause of HBF behavior.",
            "Remaining values may include other categorical state codes.",
            "Historical preprocessing and feature-selection rationale remain "
            "separate provenance questions.",
        ],
    }

    output = (
        Path(args.output_dir)
        if args.output_dir
        else processed / "v2_validation" / "raw_input_audit"
    )
    run_id = (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        + "_" + uuid.uuid4().hex[:8]
    )
    run_dir = output / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    target = run_dir / "raw_input_audit_report.json"

    with target.open("x", encoding="utf-8") as f:
        json.dump(report, f, indent=2, allow_nan=False)

    print("Read-only training-period audit completed.")
    print("Report:", target)
    print("No scientific acceptance decision was made.")


if __name__ == "__main__":
    main()