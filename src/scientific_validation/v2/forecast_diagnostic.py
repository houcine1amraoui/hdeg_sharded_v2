"""Small NumPy helpers for the fixed Level 2 forecasting diagnostic."""
import json
from pathlib import Path
import numpy as np
from src.scientific_validation.v2.preflight import require, sha256_file, validate_pair_arrays

LEVELS = ('Z', 'S', 'S_tilde', 'g')
BLOCKS = 32
ROWS = 64
BATCH = 32
SEEDS = {'train': 42, 'val': 43}


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    temporary.replace(path)


def select_blocks(n, window_size, seed):
    """Full observation footprint [s, s+ROWS+W) stays inside each stratum.

    Strata partition [0, n), the eligible pair-index range. This deliberately
    excludes the final W observation positions from the selection frame.
    """
    rng = np.random.Generator(np.random.PCG64(seed))
    edges = [i*int(n)//BLOCKS for i in range(BLOCKS+1)]
    result = []
    for block, (lo, end) in enumerate(zip(edges[:-1], edges[1:])):
        maximum = end-ROWS-window_size
        require(maximum >= lo, 'Split too short for the frozen 32 x 64 budget and window footprint')
        start = int(rng.integers(lo, maximum+1))
        result.append(dict(block=block, stratum_start=lo, stratum_end=end,
                           start_index=start, end_index=start+ROWS,
                           observation_end_exclusive=start+ROWS+window_size))
    return result


def block_shards(block, shards):
    selected = [s for s in shards if s['start_index'] < block['end_index']
                and s['end_index'] > block['start_index']]
    require(bool(selected), 'Block has no source shards')
    return selected


def read_block(block, shards, split, w, n, first_ns, step):
    """Read at most one source shard at a time; concatenate only selected rows."""
    xs, ys, starts, targets = [], [], [], []
    cursor = block['start_index']
    for item in block_shards(block, shards):
        path = Path(item['file'])
        require(sha256_file(path) == item['sha256'], f'Shard changed: {path}')
        with np.load(path, allow_pickle=False) as data:
            for key, expected in dict(split=split, shard_index=item['shard_index'],
                                      start_index=item['start_index'], end_index=item['end_index'],
                                      window_size=w, num_devices=n).items():
                require(data[key].item() == expected, f'Shard metadata mismatch: {key}')
            lo = max(block['start_index'], item['start_index'])
            hi = min(block['end_index'], item['end_index'])
            require(lo == cursor, 'Missing/duplicate/out-of-order block rows')
            sl = slice(lo-item['start_index'], hi-item['start_index'])
            shape = (item['end_index']-item['start_index'], w, n)
            for key, destination in (('X', xs), ('Y', ys)):
                array = data[key]
                require(array.shape == shape and array.dtype == np.float32, 'Invalid shard tensor')
                destination.append(array[sl].copy())
                del array
            dates = data['window_start_timestamps']
            require(dates.dtype.kind == 'M' and dates.shape == (shape[0],), 'Invalid start dates')
            require(not np.isnat(dates[sl]).any(), 'NaT start date')
            starts.append(dates[sl].astype('datetime64[ns]').astype(np.int64))
            dates = data['target_timestamps']
            require(dates.shape == shape[:2] and dates.dtype.kind == 'M', 'Invalid target dates')
            targets.append(dates[sl].copy())
        require(sha256_file(path) == item['sha256'], f'Shard changed while reading: {path}')
        cursor = hi
    require(cursor == block['end_index'], 'Incomplete block')
    x, y = np.concatenate(xs), np.concatenate(ys)
    ticks, target_dates = np.concatenate(starts), np.concatenate(targets)
    expected = first_ns + np.arange(block['start_index'], block['end_index'], dtype=np.int64)*step
    require(np.array_equal(ticks, expected), 'Selected timestamps differ from preflight timeline')
    validate_pair_arrays(x, y, ticks, target_dates, step)
    return x, y


def window_mse(prediction, target):
    a, b = np.asarray(prediction, dtype=np.float64), np.asarray(target, dtype=np.float64)
    require(a.shape == b.shape and a.ndim >= 2, 'Representation shape mismatch')
    require(np.isfinite(a).all() and np.isfinite(b).all(), 'Nonfinite representation')
    result = np.mean((a-b).reshape(len(a), -1)**2, axis=1)
    require(np.isfinite(result).all(), 'Nonfinite MSE')
    return result


def error_summary(hbf, persistence):
    h, p = float(np.mean(hbf)), float(np.mean(persistence))
    return dict(hbf_mse=h, persistence_mse=p, delta_hbf_minus_persistence=h-p,
                skill=None if p == 0 else 1-h/p,
                skill_status='undefined_zero_persistence_mse' if p == 0 else 'defined',
                skill_denominator=p)


class Moments:
    """Stable population moments; min/max retain exact row-constancy evidence."""
    def __init__(self):
        self.count = 0

    def update(self, values):
        a = np.asarray(values, dtype=np.float64).reshape(len(values), -1)
        require(len(a) > 0 and np.isfinite(a).all(), 'Invalid moment input')
        mean = a.mean(axis=0)
        m2 = ((a-mean)**2).sum(axis=0)
        low, high = a.min(axis=0), a.max(axis=0)
        if self.count == 0:
            self.mean, self.m2, self.low, self.high = mean, m2, low, high
        else:
            delta = mean-self.mean
            total = self.count+len(a)
            self.m2 += m2 + delta**2*(self.count*len(a)/total)
            self.mean += delta*(len(a)/total)
            self.low, self.high = np.minimum(self.low, low), np.maximum(self.high, high)
        self.count += len(a)

    def summary(self):
        require(self.count > 0, 'No representation samples')
        # A repeated floating-point value is exactly constant even if mean rounding
        # creates a tiny numerical residual in a centered sum of squares.
        m2 = np.where(self.low == self.high, 0.0, self.m2)
        return dict(count=self.count, coordinates=len(self.mean),
                    centered_population_variance=float(np.mean(m2/self.count)),
                    mean_vector_l2_norm=float(np.linalg.norm(self.mean)),
                    exactly_constant_across_windows=bool(np.array_equal(self.low, self.high)),
                    fraction_exactly_constant_coordinates=float(np.mean(self.low == self.high)))
