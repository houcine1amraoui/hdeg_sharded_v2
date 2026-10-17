"""Dependency-light synthetic tests; no HDEG data or trained model required.
Run from project root: python tests/test_r3_pilot.py
"""
from pathlib import Path
import importlib.util
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("pilot_helpers", ROOT / "src/scientific_validation/v1/r3_pilot.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def rejected(call):
    try:
        call()
    except (ValueError, FloatingPointError):
        return
    raise AssertionError("Invalid input accepted")


def reference(x, y):
    x, y = x.reshape(-1, x.shape[-1]), y.reshape(-1, y.shape[-1])
    value = (2 * np.linalg.norm(x[:, None] - y[None, :], axis=2).mean()
             - np.linalg.norm(x[:, None] - x[None, :], axis=2).mean()
             - np.linalg.norm(y[:, None] - y[None, :], axis=2).mean())
    return np.sqrt(max(0., value))


def main():
    rng = np.random.default_rng(5)
    for tile in (1, 3, 128):
        x, y = rng.normal(size=(4, 5, 3)), rng.normal(size=(4, 5, 3))
        xx = m.block_distance_means(x, x, same=True, tile_size=tile)
        xy = m.block_distance_means(x, y, tile_size=tile)
        yy = m.block_distance_means(y, y, same=True, tile_size=tile)
        assert np.all(np.diag(xx) > 0), "Block diagonal must retain within-block distances"
        for _ in range(30):
            ix, iy = rng.integers(4, size=4), rng.integers(4, size=4)
            w, v = np.bincount(ix, minlength=4) / 4, np.bincount(iy, minlength=4) / 4
            np.testing.assert_allclose(m.energy_from_weights(xx, xy, yy, w, v),
                                       reference(x[ix], y[iy]), atol=1e-10, rtol=1e-10)
        np.testing.assert_allclose(m.energy_from_weights(xx, xx, xx, np.ones(4)/4, np.ones(4)/4), 0, atol=1e-10)
    constant = np.ones((3, 4, 2)) * 1e8
    assert np.all(m.block_distance_means(constant, constant, same=True) == 0)
    w = m.make_resampling_weights(4, 100, 1042)
    np.testing.assert_array_equal(w, m.make_resampling_weights(4, 100, 1042))
    np.testing.assert_allclose(w.sum(axis=1), 1)
    ts = np.datetime64('2022-01-01', 'ns') + np.arange(43) * np.timedelta64(1, 's')
    ts[20:] += np.timedelta64(10, 's')
    kw = dict(step_ns=10**9, block_length=4, primary_blocks=8, seed=42)
    plan, indices = m.make_block_plan(ts, pilot_blocks=2, **kw)
    full, all_indices = m.make_block_plan(ts, pilot_blocks=8, **kw)
    assert plan['gap_count'] == 1 and plan['excluded_tail_rows'] == 3
    np.testing.assert_array_equal(indices, all_indices[:8])
    assert plan['primary_blocks'] == full['primary_blocks']
    for block in full['primary_blocks']:
        assert not block['start_index'] < 20 < block['end_index']
    assert len(np.unique(all_indices)) == len(all_indices)
    guarded, _ = m.make_block_plan(ts, pilot_blocks=2, window_size=3, **kw)
    assert guarded['excluded_gap_crossing_window_rows'] == 2
    assert all(b['end_index'] <= 18 or b['start_index'] >= 20
               for b in guarded['primary_blocks'])
    rejected(lambda: m.make_block_plan(ts[::-1], pilot_blocks=2, **kw))
    rejected(lambda: m.make_block_plan(ts[:8], pilot_blocks=2, **kw))
    rejected(lambda: m.make_block_plan(ts.astype('int64'), pilot_blocks=2, **kw))
    rejected(lambda: m.block_distance_means(x, y, same=True))
    rejected(lambda: m.block_distance_means(x * np.nan, y))
    rejected(lambda: m.energy_from_weights(np.ones((2,2))*10, np.zeros((2,2)),
                                           np.ones((2,2))*10, np.ones(2)/2, np.ones(2)/2))
    rejected(lambda: m.positive_int(True, 'budget'))
    print('PASS: 90 exact resampling comparisons; within-block diagonal; identity; constants; weights; gaps; primary/pilot nesting; invalid inputs.')


if __name__ == '__main__':
    main()
