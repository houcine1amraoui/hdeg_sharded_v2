"""Run from the project root: python tests/test_r3_primary.py (NumPy only)."""
import copy
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.scientific_validation.v1.r3_pilot import (
    CONDITIONS, LEVELS, make_block_plan, make_resampling_weights,
    block_distance_means, energy_from_weights, summarize_resampling)
from src.scientific_validation.v1.r3_primary import (
    validate_pilot, configuration_hash, frozen_primary_indices)


def rejected(call):
    try:
        call()
    except ValueError:
        return
    raise AssertionError("Invalid input accepted")


def main():
    ts = np.datetime64("2022-01-01", "ns") + np.arange(6000) * np.timedelta64(1, "s")
    # Exercise an internal gap and its W-1 excluded input windows.
    ts[3000:] += np.timedelta64(100, "s")
    plans = {}
    for k, c in enumerate(CONDITIONS):
        plan, pilot_rows = make_block_plan(ts, step_ns=10**9, block_length=128,
                                          pilot_blocks=8, primary_blocks=32,
                                          seed=42+k, window_size=30)
        plans[c] = plan
        indices = frozen_primary_indices(plan, ts, step_ns=10**9, seed=42+k, window_size=30)
        assert len(indices) == len(np.unique(indices)) == 4096
        np.testing.assert_array_equal(indices[:1024], pilot_rows)
        assert np.all(np.diff(indices.reshape(32, 128), axis=1) == 1)
        assert not np.isin(indices, np.arange(2971, 3000)).any()
        altered = copy.deepcopy(plan)
        altered["primary_blocks"][-1]["start_index"] += 1
        rejected(lambda: frozen_primary_indices(altered, ts, step_ns=10**9,
                                                seed=42+k, window_size=30))
        changed_ts = ts.copy(); changed_ts += np.timedelta64(1, "s")
        rejected(lambda: frozen_primary_indices(plan, changed_ts, step_ns=10**9,
                                                seed=42+k, window_size=30))
    config = {"seed": 42, "project_root_dir": "/old", "v1_validation": {},
              "hdeg": {"hbf": {"checkpoint": "epoch_0001.pt"}}}
    report = dict(status="COMPLETE", dataset="CU", levels=dict.fromkeys(LEVELS),
                  configuration=config, configuration_sha256=configuration_hash(config),
                  block_plans=plans, settings=dict(block_length_windows=128,
                  pilot_blocks_per_condition=8, primary_blocks_per_condition=32,
                  resampling_replicates=100, selection_seed=42, resampling_seed=1042))
    validate_pilot(report, plans, config, "CU")
    relocated = copy.deepcopy(config)
    relocated["project_root_dir"] = "/new"
    relocated["v1_validation"]["r3-primary"] = {"tile_size": 64}
    validate_pilot(report, plans, relocated, "CU")
    changed = copy.deepcopy(config); changed["seed"] = 43
    rejected(lambda: validate_pilot(report, plans, changed, "CU"))
    rejected(lambda: validate_pilot(dict(report, status="RUNNING"), plans, config, "CU"))
    rejected(lambda: validate_pilot(report, plans, config, "BRE"))
    altered = copy.deepcopy(plans); altered["N1"]["selection_seed"] = 9
    rejected(lambda: validate_pilot(report, altered, config, "CU"))
    # Compare cached summaries to the fully expanded 4096-row energy V-statistic.
    # Constant coordinates within each block make the reference exact and cheap;
    # nonconstant blocks are covered by test_r3_pilot.py's 90 comparisons.
    x = np.arange(32, dtype=float)[:, None, None] * np.ones((1, 128, 1))
    y = x + 0.5
    xx = block_distance_means(x, x, same=True)
    xy = block_distance_means(x, y)
    yy = block_distance_means(y, y, same=True)
    wx, wy = (make_resampling_weights(32, 1000, seed) for seed in (1042, 1043))
    np.testing.assert_array_equal(wx, make_resampling_weights(32, 1000, 1042))
    np.testing.assert_allclose(wx.sum(axis=1), 1)
    result, draws = summarize_resampling(xx, xy, yy, wx, wy)
    assert draws.shape == (1000,) and np.isfinite(draws).all()
    for i in (0, 500, 999):
        ix = np.repeat(np.arange(32), np.rint(wx[i] * 32).astype(int))
        iy = np.repeat(np.arange(32), np.rint(wy[i] * 32).astype(int))
        a, b = x[ix, 0, 0], y[iy, 0, 0]
        reference = np.sqrt(max(0, 2*np.abs(a[:, None]-b).mean()
                                -np.abs(a[:, None]-a).mean()-np.abs(b[:, None]-b).mean()))
        np.testing.assert_allclose(draws[i], reference, atol=1e-12)
    assert result["population_ci_status"] == "not_established"
    print("PASS: frozen selection, gap guards, provenance/config rejection, 32 x 128 budget, 1000 resamples.")


if __name__ == "__main__":
    main()
