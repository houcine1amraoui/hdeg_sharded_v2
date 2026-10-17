"""Pure NumPy helpers for the standalone, descriptive R3 computational pilot."""
from __future__ import annotations
import hashlib
from numbers import Integral
import numpy as np

CONDITIONS = {"N1": "val", "A2": "actor2_test", "N2": "actor1_test"}
LEVELS = ("Z", "S", "S_tilde", "g")
CONTRASTS = (("N1", "A2"), ("A2", "N2"), ("N1", "N2"))


def positive_int(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def index_hash(indices):
    return hashlib.sha256(np.asarray(indices, dtype="<i8").tobytes()).hexdigest()


def make_block_plan(timestamps, *, step_ns, block_length, pilot_blocks,
                    primary_blocks, seed, window_size=1):
    """Partition true time segments; sample primary blocks once, then pilot prefix.

    The plan's block order is the seeded draw order. It must be retained while
    loading and computing summaries. Adjacent storage shards are not segments.
    """
    step_ns = positive_int(step_ns, "step_ns")
    length = positive_int(block_length, "block_length")
    count = positive_int(pilot_blocks, "pilot_blocks")
    primary = positive_int(primary_blocks, "primary_blocks")
    if count > primary:
        raise ValueError("pilot_blocks cannot exceed primary_blocks")
    window_size = positive_int(window_size, "window_size")
    raw = np.asarray(timestamps)
    if raw.ndim != 1 or raw.dtype.kind not in "MUS":
        raise ValueError("Window timestamps must be 1-D datetime64 or timestamp strings")
    ts = raw.astype("datetime64[ns]")
    if len(ts) < 2 or np.isnat(ts).any():
        raise ValueError("At least two valid window timestamps are required")
    ticks = ts.astype(np.int64)
    delta = np.diff(ticks)
    if np.any(delta <= 0):
        raise ValueError("Timestamps must be strictly increasing, including across shards")
    if np.any(delta < step_ns) or np.any(delta % step_ns):
        raise ValueError("Timestamp spacing conflicts with configured sampling interval")
    cuts = np.r_[0, np.flatnonzero(delta != step_ns) + 1, len(ts)]
    starts, segments = [], []
    for lo, hi in zip(cuts[:-1], cuts[1:]):
        lo, hi = int(lo), int(hi)
        # Near an internal gap, the last W-1 input windows cross that gap.
        safe_hi = max(lo, hi - (window_size - 1)) if hi < len(ts) else hi
        nfull = (safe_hi - lo) // length
        starts.extend(range(lo, lo + nfull * length, length))
        segments.append({"start_index": lo, "end_index": hi,
                         "candidate_blocks": nfull,
                         "gap_crossing_window_rows": hi - safe_hi,
                         "excluded_tail_rows": safe_hi - lo - nfull * length})
    starts = np.asarray(starts, dtype=np.int64)
    if len(starts) < primary:
        raise ValueError(f"Need {primary} eligible blocks; found {len(starts)}. No automatic budget reduction.")
    ids = np.random.default_rng(seed).choice(len(starts), size=primary, replace=False)
    selected = starts[ids]
    def records(ids_, starts_):
        return [{"candidate_id": int(i), "start_index": int(s),
                 "end_index": int(s + length), "start_timestamp": str(ts[s]),
                 "last_timestamp": str(ts[s + length - 1])}
                for i, s in zip(ids_, starts_)]
    indices = (selected[:count, None] + np.arange(length)).reshape(-1)
    return {"selection_seed": int(seed), "population_rows": len(ts),
            "timestamp_sha256": index_hash(ticks), "step_ns": step_ns,
            "gap_count": len(segments) - 1, "segments": segments,
            "block_length_windows": length, "candidate_blocks": len(starts),
            "excluded_tail_rows": sum(s["excluded_tail_rows"] for s in segments),
            "excluded_gap_crossing_window_rows": sum(s["gap_crossing_window_rows"] for s in segments),
            "primary_blocks": records(ids, selected),
            "pilot_blocks": records(ids[:count], selected[:count]),
            "pilot_index_sha256": index_hash(indices)}, indices


def block_distance_means(x, y, *, tile_size=128, same=False, progress=None):
    """Exact mean Euclidean distance per block pair; no distance of block means.

    Inputs: (blocks, windows, coordinates). Coordinate vectorization stays within
    a hierarchy level. Keep nonzero within-block off-observation distances.
    """
    tile = positive_int(tile_size, "tile_size")
    if tile > 512:
        raise ValueError("tile_size must not exceed 512 in this pilot")
    if same and y is not x:
        raise ValueError("same=True requires the identical input array")
    a, b = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    if a.ndim != 3 or b.ndim != 3 or min(a.shape) == 0 or min(b.shape) == 0:
        raise ValueError("Expected nonempty (blocks, windows, coordinates) arrays")
    if a.shape[2] != b.shape[2] or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Nonfinite values or incompatible coordinate dimensions")
    # One common translation improves numerical stability; distances unchanged.
    origin = a[0, 0].copy()
    result = np.empty((len(a), len(b)), dtype=np.float64)
    for p in range(len(a)):
        for q in range(p if same else 0, len(b)):
            total = 0.0
            for i in range(0, a.shape[1], tile):
                ai = a[p, i:i + tile] - origin
                for j in range(0, b.shape[1], tile):
                    bj = b[q, j:j + tile] - origin
                    squared = (np.sum(ai * ai, axis=1)[:, None]
                               + np.sum(bj * bj, axis=1)[None, :] - 2 * (ai @ bj.T))
                    np.maximum(squared, 0, out=squared)
                    if same and p == q:
                        rows = np.arange(i, i + len(ai))[:, None]
                        cols = np.arange(j, j + len(bj))[None, :]
                        squared[rows == cols] = 0
                    total += np.sqrt(squared).sum(dtype=np.float64)
            result[p, q] = total / (a.shape[1] * b.shape[1])
            if same:
                result[q, p] = result[p, q]
        if progress:
            progress(p + 1, len(a))
    return result


def energy_from_weights(kxx, kxy, kyy, w, v):
    w, v = np.asarray(w, dtype=np.float64), np.asarray(v, dtype=np.float64)
    if (w.ndim != 1 or v.ndim != 1 or not np.isfinite(w).all()
            or not np.isfinite(v).all() or np.any(w < 0) or np.any(v < 0)
            or not np.isclose(w.sum(), 1) or not np.isclose(v.sum(), 1)):
        raise ValueError("Block weights must be finite, nonnegative and sum to one")
    kxx, kxy, kyy = map(np.asarray, (kxx, kxy, kyy))
    if (kxx.shape != (len(w), len(w)) or kxy.shape != (len(w), len(v))
            or kyy.shape != (len(v), len(v))):
        raise ValueError("Block-summary shapes do not match weights")
    if any(not np.isfinite(k).all() or np.any(k < 0) for k in (kxx, kxy, kyy)):
        raise ValueError("Invalid block distances")
    cross, within_x, within_y = 2 * w @ kxy @ v, w @ kxx @ w, v @ kyy @ v
    squared = cross - within_x - within_y
    tolerance = 1e-12 * max(1.0, abs(cross), abs(within_x), abs(within_y))
    if squared < -tolerance:
        raise FloatingPointError(f"Materially negative squared energy: {squared}")
    return float(np.sqrt(max(0.0, squared)))


def make_resampling_weights(block_count, replicates, seed):
    n = positive_int(block_count, "block_count")
    r = positive_int(replicates, "replicates")
    return np.random.default_rng(seed).multinomial(n, np.full(n, 1 / n), size=r) / n


def summarize_resampling(kxx, kxy, kyy, weights_x, weights_y):
    if weights_x.shape[0] != weights_y.shape[0] or len(weights_x) < 2:
        raise ValueError("At least two paired resampling replicates are required")
    point = energy_from_weights(kxx, kxy, kyy,
                                np.full(len(kxx), 1 / len(kxx)),
                                np.full(len(kyy), 1 / len(kyy)))
    values = np.array([energy_from_weights(kxx, kxy, kyy, w, v)
                       for w, v in zip(weights_x, weights_y)])
    return {"point_estimate": point, "resampling_sd": float(values.std(ddof=1)),
            "resampling_q025": float(np.quantile(values, .025)),
            "resampling_q975": float(np.quantile(values, .975)),
            "replicate_count": len(values), "population_ci_status": "not_established",
            "uncertainty_scope": "conditional_selected_block_resampling",
            "pilot_only": True}, values
