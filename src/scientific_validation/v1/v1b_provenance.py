"""Provenance helpers for V1-B evidence."""

from __future__ import annotations

from pathlib import Path
from numbers import Integral
import hashlib


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, Integral) or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
    path = Path(path)
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def file_manifest(path: Path) -> dict:
    path = Path(path)
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def load_config_yaml(path: Path) -> dict:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to load config.yaml") from exc
    with Path(path).open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ValueError("Configuration YAML must contain a mapping")
    return config


def build_protocol_manifest(
    *,
    dataset: str,
    seed: int,
    window_size: int,
    num_devices: int,
    representation_dim: int,
    r2_pair_count: int,
    r3_max_per_condition: int,
    r3_block_length_windows: int,
    bootstrap_replicates: int,
) -> dict:
    # This records requested settings, never certifies completed computation.
    for name, value in (
        ("seed", seed), ("window_size", window_size), ("num_devices", num_devices),
        ("representation_dim", representation_dim), ("r2_pair_count", r2_pair_count),
        ("r3_max_per_condition", r3_max_per_condition),
        ("r3_block_length_windows", r3_block_length_windows),
        ("bootstrap_replicates", bootstrap_replicates),
    ):
        minimum = 0 if name in ("seed", "bootstrap_replicates") else 1
        if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    return {
        "manifest_kind": "requested_protocol_settings",
        "execution_status": "not_attested_by_this_manifest",
        "dataset": dataset,
        "seed": seed,
        "window_size": window_size,
        "num_devices": num_devices,
        "representation_dim": representation_dim,
        "r2": {
            "pair_count": r2_pair_count,
            "sampling": "deterministic_representation_blind_without_replacement",
        },
        "r3": {
            "max_per_condition": r3_max_per_condition,
            "sampling": "deterministic_representation_blind_without_replacement",
            "condition_seeds": {"N1": int(seed), "A2": int(seed) + 1, "N2": int(seed) + 2},
            "uncertainty": {
                "method": "moving_block_bootstrap_only_for_complete_temporal_sequences",
                "block_length_windows": r3_block_length_windows,
                "requested_bootstrap_replicates": bootstrap_replicates,
                "sparse_sample_policy": "withhold_temporal_confidence_interval",
                "eligibility": "complete chronological populations; block length smaller than both populations",
                "execution_status": "consult_per_comparison_result",
                "interval_method": "percentile",
                "confidence_level": 0.95,
            },
        },
        "conditions": {
            "N1": "val",
            "A2": "actor2_test",
            "N2": "actor1_test",
        },
    }
