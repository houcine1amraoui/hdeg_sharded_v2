"""Validation and deterministic plans for descriptive R3 sensitivity runs."""
import copy
from .r3_pilot import CONDITIONS, LEVELS, CONTRASTS, make_block_plan, index_hash
from .r3_primary import configuration_hash


def validate_primary(report, plans, config, dataset):
    if report.get("status") != "COMPLETE" or report.get("pilot_only") is not False:
        raise ValueError("A completed primary R3 report is required")
    if report.get("dataset") != dataset or set(report.get("levels", {})) != set(LEVELS):
        raise ValueError("Primary dataset/representation coverage mismatch")
    if set(plans) != set(CONDITIONS) or plans != report.get("block_plans"):
        raise ValueError("Primary block plan differs from its report")
    if configuration_hash(report["configuration"]) != report.get("configuration_sha256"):
        raise ValueError("Primary configuration hash mismatch")
    def comparable(value):
        value = copy.deepcopy(value)
        value.pop("project_root_dir", None)
        value.get("v1_validation", {}).pop("r3-sensitivity", None)
        return value
    if comparable(config) != comparable(report["configuration"]):
        raise ValueError("Configuration changed since primary (apart from root/r3-sensitivity)")
    expected = dict(block_length_windows=128, blocks_per_condition=32,
                    selected_rows_per_condition=4096, resampling_replicates=1000,
                    selection_seed=config["seed"], resampling_seed=config["seed"]+1000)
    if any(report["settings"].get(k) != v for k, v in expected.items()):
        raise ValueError("Primary budget or seeds mismatch")
    for level in LEVELS:
        results = report["levels"][level]["results"]
        if set(results) != {f"{a}-{b}" for a, b in CONTRASTS}:
            raise ValueError("Incomplete primary contrasts")
        if any(r.get("replicate_count") != 1000 or r.get("analysis_role") != "primary_descriptive"
               for r in results.values()):
            raise ValueError("Primary result role/replicate mismatch")


def sensitivity_plan(timestamps, *, length, step_ns, seed, window_size):
    if isinstance(length, bool) or length not in (64, 256):
        raise ValueError("Sensitivity block length must be 64 or 256")
    plan, indices = make_block_plan(timestamps, step_ns=step_ns,
        block_length=length, pilot_blocks=32, primary_blocks=32,
        seed=seed, window_size=window_size)
    # Reuse the same tested partitioning algorithm without misleading pilot labels.
    plan["selected_blocks"] = plan.pop("primary_blocks")
    plan.pop("pilot_blocks")
    plan.pop("pilot_index_sha256")
    plan["selected_index_sha256"] = index_hash(indices)
    plan["analysis_role"] = "block_length_sensitivity"
    plan["selection_scope"] = "length_specific_partition_uniform_without_replacement"
    return plan, indices
