# HDEG Level 1 R3: computational and resampling decision

Status: design for the next focused implementation, not an executed experiment.
The existing run_v1b.py and configuration have not been changed by this document.

## Decision and scope

Use exact empirical energy distance on a bounded sample of original contiguous
blocks, with cached block-pair distance summaries. Resample those intact blocks
for a descriptive temporal-block resampling interval. Do not call the resulting
2.5–97.5% range a calibrated 95% population confidence interval.

This is a deliberate change from uniform individual-row sampling. The target
remains a within-level distribution contrast over the recorded condition, but
the finite-sample estimator uses a different sampling design. It is not ordinary
moving-block bootstrap applied to the complete recorded series. No claim of
nominal coverage, hypothesis-test significance, or generalization to residents
is attached to this sampled-block resampling procedure.

The previous bootstrap correction stays in force until this design has its own
implementation and tests. Do not enable the old sparse-row bootstrap.

## Fixed scientific quantities

- Conditions: N1=val (181,410 rows), A2=actor2_test (259,170),
  N2=actor1_test (597,557).
- Contrasts: N1–A2, A2–N2, N1–N2, all reported.
- Levels: Z, S, S_tilde, g, separately in native coordinates.
- Metric: sqrt(2 mean cross-distance − mean within-X distance − mean within-Y
  distance), with V-statistic denominators and zero observation diagonals.
- Use all distances between the selected observations, not distances between
  block means. No PCA, level concatenation, or new normalization is introduced.
- Checkpoint fixed to the successfully audited epoch_0001.pt identity.
- R3 measures actor/period contrasts, not labeled anomaly discrimination.

## Budget

| Run | Blocks per condition | Windows per block | Windows per condition | Resampling replicates |
|---|---:|---:|---:|---:|
| Computational pilot | 8 | 128 | 1,024 | 100, verification only |
| Primary descriptive run | 32 | 128 | 4,096 | 1,000 |
| Short-block sensitivity | 32 | 64 | 2,048 | 1,000 |
| Long-block sensitivity | 32 | 256 | 8,192 | 1,000 |

Sensitivity settings are prespecified. They must not be selected retrospectively
for larger separation or narrower intervals. Holding block count fixed changes
row count with length: explicitly report this confounding. If conclusions depend
on these settings, report sensitivity rather than claiming robustness. A later
matched-row-count check can isolate that issue if needed; do not add it silently.

128 windows is a practical starting resolution, not an estimated optimal block
length. The supplied configuration uses 1-second sampling and W=30. A 128-row
block spans 127 seconds of window-start times and approximately 157 seconds of
raw inputs including overlap. Verify actual timestamps before attaching units.
Block lengths 64 and 256 likewise refer to original adjacent window indices.

For three conditions, cache three within-condition and three cross-condition
summaries per level. Without exploiting symmetry, construction requires
6 n^2 pair-distance evaluations per level, or 24 n^2 over four levels:

- Pilot: 25,165,824 pair distances.
- Primary: 402,653,184.
- Long-block sensitivity: 1,610,612,736.
- A 50,000-row design with the same reuse: 60,000,000,000.

Thus the primary sample reduces pair enumeration by about 149-fold relative to
50,000 rows, before removing repeated distance calculations from bootstrap.
These are operation counts, not measured wall-clock predictions. Per-distance
cost differs by level dimension; Z has 1,472 coordinates and g has 64.

Use float64 accumulation and bounded tiles (at most 512 x 512). At most two
selected conditions and one level need be resident. Two 4,096-row Z arrays in
float64 occupy about 92 MiB, excluding input shards and working buffers. Cache
only 32 x 32 block-pair summaries, not dense 4,096 x 4,096 distance matrices.
Cache within-condition summaries for reuse across contrasts.

The pilot records load time, distance-computation time, resampling time, peak
resident memory, NumPy/BLAS versions, thread count, and hardware. Forecast the
primary compute cost from the 16-fold n^2 increase, separately from I/O. Do not
promise a runtime without this measurement. On interruption, retain a clearly
marked incomplete checkpoint; never silently shrink the sample budget.

## Original-time sampling

1. Read global split-local indices and window_start_timestamps. Validate regular
   spacing; a storage shard boundary is not a temporal discontinuity.
2. Partition each genuinely contiguous temporal segment into non-overlapping
   candidate blocks of L consecutive rows. Never bridge a gap or a condition
   boundary. Blocks may cross storage shards when timestamps are continuous.
3. Uniformly sample 32 candidate blocks without replacement per condition using
   seeds 42, 43, 44 for N1, A2, N2. Sort candidates for reproducible loading only.
   Reuse the same selected blocks at every level and every contrast.
4. Retain every observation in a selected block. Never regard the join between
   two nonadjacent selected blocks as a real consecutive-time transition.
5. Keep complete equal-length blocks; report every excluded tail and any gap.
   With a continuous series and L=128, excluded tails are 34, 98, and 53 rows,
   respectively. Gaps may change these counts, so calculate them from timestamps.
6. Save block identifiers, original ranges, timestamps, seeds, eligible population
   sizes, and checksums. Pilot blocks should be a prefix of the same seeded
   candidate draw, with the primary sample frozen before evaluating contrasts.

## Resampling and exact computational shortcut

For each condition independently, draw 32 block IDs with replacement from its
32 selected blocks. Use one reproducible weight vector per condition per
replicate, shared across levels and contrasts. This preserves within-block
observations jointly across every device/state/embedding coordinate.

Use bootstrap base seed 1042 (separate from selection), with condition offsets
0/1/2; record these values. Bootstrap randomness must never change which source
blocks form the primary selected population.

Let K_XY[p,q] be the mean distance between all L^2 observation pairs in selected
blocks p and q. With normalized resampling counts w and v, calculate:

    squared_energy = 2 w^T K_XY v − w^T K_XX w − v^T K_YY v
    energy = sqrt(squared_energy)

Uniform weights give the primary point estimate. The formula exactly equals
explicitly duplicating and concatenating resampled blocks for this empirical
V-statistic. The K_XX block diagonal is generally NOT zero: it contains distances
between different observations in the same block. Only individual self-pair
entries are zero. Tolerance-clamp tiny negative roundoff and reject material
negative values, as in the existing metric.

This identity was checked against explicit expanded-array calculations for
100 synthetic resamples at absolute/relative tolerance 1e-12. It verifies the
algebra, not confidence-interval coverage.

After constructing the six block-summary matrices for a level, 1,000 resamples
require only small quadratic forms. Do not recompute raw distances or reload
representation shards for each replicate.

## Reporting uncertainty honestly

Report point_estimate, resampling_sd, resampling_q025, resampling_q975,
replicate_count, block_length_windows, selected_block_count, selection_seed,
resampling_seed, and uncertainty_scope=conditional_selected_block_resampling.
Set population_ci_status=not_established. Keep this schema distinct from ci_lower
and ci_upper used for inferential confidence intervals.

Interpret the central 95% resampling range as variation induced by reweighting
the selected temporal blocks. It does not quantify all uncertainty from omitted
blocks, long-term behavior, training seeds, checkpoint selection, or residents.
Independently resampling N1 and N2 does not model dependence between periods of
the same resident. These limits remain even when the range is narrow.

Inspect time coverage, hour-of-day coverage, block-level summaries, and lag
dependence before interpreting results. These diagnostics do not prove that
128 windows captures all dependence. If longer-range patterns or block-selection
sensitivity dominate, retain descriptive interpretation and revise a future
inferential protocol explicitly. Never infer significance from exclusion of zero:
finite-sample energy is nonnegative and the interval is not a null test.

For calibrated population confidence intervals, further work is required on the
estimand, dependence/stationarity assumptions, cross-period dependence, block
selection, and estimator-specific coverage simulations. This document deliberately
settles an executable descriptive R3 procedure without claiming that work is done.

## Next implementation boundary

Implement only R3's block sampler, timestamp validation, cached distance summaries,
resampling weights, and a standalone computational pilot runner. Keep R1/R2,
training, inference artifacts, and Levels 2–4 unchanged. Preserve YAML configuration
loading. New R3 block settings must be explicit; r3-max=50000 must not be silently
reinterpreted. First deliver and run the pilot, then inspect its timing/diagnostics
before the primary experiment.

## Methodological references

- Székely and Rizzo (2013), Energy statistics: A class of statistics based on
  distances. https://doi.org/10.1016/j.jspi.2013.03.018
  Supports the distance-based U/V-statistic formulation, not this selected-block
  procedure's nominal coverage.
- Ju, Moving Block Bootstrap for Analyzing Longitudinal Data.
  https://pmc.ncbi.nlm.nih.gov/articles/PMC4443935/
  Discusses dependence preservation and block-size limitations. The present
  selected-block resampling design is not that paper's regression bootstrap.
- Hesterberg, What Teachers Should Know about the Bootstrap.
  https://arxiv.org/abs/1411.5279
  Discusses limitations of percentile confidence intervals; quantile labels alone
  do not establish coverage.
