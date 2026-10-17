"""Frozen-input checks for the standalone primary R3 run (pure NumPy)."""
import copy
import hashlib
import json
import numpy as np

from .r3_pilot import CONDITIONS, LEVELS, index_hash, make_block_plan


def configuration_hash(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()


def validate_pilot(report, plans, config, dataset):
    if report.get("status") != "COMPLETE" or set(report.get("levels", {})) != set(LEVELS):
        raise ValueError("A completed pilot covering all four representations is required")
    if report.get("dataset") != dataset:
        raise ValueError("Pilot dataset differs from the current dataset")
    if set(plans) != set(CONDITIONS) or report.get("block_plans") != plans:
        raise ValueError("Frozen block plan differs from the completed pilot report")
    old = report["configuration"]
    if configuration_hash(old) != report.get("configuration_sha256"):
        raise ValueError("Pilot configuration hash mismatch")
    # Relocating the project and setting the new runner's execution options is
    # allowed. All scientific, preprocessing and checkpoint settings stay fixed.
    def comparable(value):
        value = copy.deepcopy(value)
        value.pop("project_root_dir", None)
        value.get("v1_validation", {}).pop("r3-primary", None)
        return value
    if comparable(old) != comparable(config):
        raise ValueError("Configuration changed since pilot (apart from project root/r3-primary)")
    seed = config["seed"]
    settings = report["settings"]
    expected = dict(block_length_windows=128, pilot_blocks_per_condition=8,
                    primary_blocks_per_condition=32, resampling_replicates=100,
                    selection_seed=seed, resampling_seed=seed + 1000)
    if any(settings.get(k) != v for k, v in expected.items()):
        raise ValueError("Pilot does not match the frozen 8/32 blocks x 128 protocol")


def frozen_primary_indices(plan, timestamps, *, step_ns, seed, window_size):
    # Reproduction is a validation only. Selection below comes from the saved
    # primary_blocks, never from a new or automatically adjusted draw.
    reproduced, _ = make_block_plan(
        timestamps, step_ns=step_ns, block_length=128, pilot_blocks=8,
        primary_blocks=32, seed=seed, window_size=window_size)
    if reproduced != plan:
        raise ValueError("Frozen plan no longer matches timestamps, seed or block eligibility")
    starts = np.array([b["start_index"] for b in plan["primary_blocks"]], dtype=np.int64)
    indices = (starts[:, None] + np.arange(128)).reshape(-1)
    if len(indices) != 4096 or len(np.unique(indices)) != 4096:
        raise ValueError("Primary selection must contain 4096 distinct window rows")
    if index_hash(indices[:1024]) != plan["pilot_index_sha256"]:
        raise ValueError("Primary prefix differs from the pilot selection")
    return indices
