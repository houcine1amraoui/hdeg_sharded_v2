"""Deterministic, representation-blind V1-B sampling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict
from numbers import Integral
import numpy as np


CONDITION_ORDER = ("N1", "A2", "N2")


def _nonnegative_integer(value, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    value = int(value)
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


@dataclass(frozen=True)
class Pair:
    i: int
    j: int


def deterministic_pairs(
    n: int,
    count: int = 100_000,
    seed: int = 42,
) -> np.ndarray:
    """Deterministically sample unordered sample-index pairs without replacement.

    Pair selection depends only on population size, seed and pair count.
    It never inspects representation values.
    """
    n = _nonnegative_integer(n, "n")
    count = _nonnegative_integer(count, "count")
    seed = _nonnegative_integer(seed, "seed")
    if n > np.iinfo(np.int64).max:
        raise ValueError("Population exceeds int64 index range")
    if n < 2:
        return np.empty((0, 2), dtype=np.int64)

    total = n * (n - 1) // 2
    if total > np.iinfo(np.int64).max:
        raise ValueError("Unordered pair population exceeds int64 rank range")
    m = min(count, total)

    rng = np.random.default_rng(seed)

    # For the normal CU case total >> m. We sample encoded pair ranks.
    ranks = rng.choice(total, size=m, replace=False)
    ranks.sort()

    pairs = np.empty((m, 2), dtype=np.int64)

    # Decode a lexicographic upper-triangle rank.
    # row i contains pairs (i,j), j>i, of length n-i-1.
    for k, r in enumerate(ranks):
        r = int(r)
        lo, hi = 0, n - 2
        while lo <= hi:
            mid = (lo + hi) // 2
            # number of pairs before row mid
            before = mid * (2 * n - mid - 1) // 2
            if before <= r:
                lo = mid + 1
            else:
                hi = mid - 1
        i = max(0, lo - 1)
        before = i * (2 * n - i - 1) // 2
        j = i + 1 + (r - before)
        pairs[k] = (i, j)

    return pairs


def deterministic_sample_indices(
    n: int,
    count: int,
    seed: int = 42,
) -> np.ndarray:
    """Deterministic without-replacement sample of population indices."""
    n = _nonnegative_integer(n, "n")
    count = _nonnegative_integer(count, "count")
    seed = _nonnegative_integer(seed, "seed")
    if n > np.iinfo(np.int64).max:
        raise ValueError("Population exceeds int64 index range")
    if n == 0:
        return np.empty(0, dtype=np.int64)
    m = min(int(count), n)
    rng = np.random.default_rng(seed)
    out = rng.choice(n, size=m, replace=False)
    out.sort()
    return out


def condition_sampling_plan(
    populations: Dict[str, int],
    max_per_condition: int = 50_000,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """Use the engine's N1/A2/N2 seed assignment, independent of dict order.

    For other condition names, retain deterministic alphabetical ordering.
    Sorted returned row indices preserve chronology but not temporal adjacency.
    """
    max_per_condition = _nonnegative_integer(max_per_condition, "max_per_condition")
    seed = _nonnegative_integer(seed, "seed")
    if any(not isinstance(name, str) for name in populations):
        raise TypeError("Condition names must be strings")
    if set(populations).issubset(CONDITION_ORDER):
        # Keep offsets stable even when a subset of HDEG conditions is requested.
        ordered = [(k, name) for k, name in enumerate(CONDITION_ORDER)
                   if name in populations]
    else:
        ordered = list(enumerate(sorted(populations)))
    return {
        name: deterministic_sample_indices(
            populations[name], max_per_condition, seed=seed + k
        )
        for k, name in ordered
    }
