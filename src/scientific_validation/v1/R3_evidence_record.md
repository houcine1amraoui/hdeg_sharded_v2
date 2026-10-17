# HDEG Level 1 — R3 evidence record

Status: primary and prespecified sensitivity runs completed; saved-summary numerical audit passed. This is not a declaration that Level 1 is complete.

## Protocol and provenance

CU; checkpoint epoch_0001.pt; SHA256 `d47f7b28843aac65961154f52a0f12da83e6bd944cc2da5b3d22bab02755cf81`. Dimensions N=23, K=9, D=64; input window W=30 at a one-second window-start interval. Conditions: N1=val (Actor 1 earlier period), A2=actor2_test, N2=actor1_test (Actor 1 later period).

Primary: 32 blocks x 128 consecutive windows per condition; 1000 resampling replicates. Sensitivities: 32 blocks x 64 and 32 blocks x 256, also 1000 replicates. Selection seeds 42/43/44 and resampling seeds 1042/1043/1044. Each sensitivity uses a length-specific partition and draw, not the identical physical blocks.

All representations share each run’s selection and resampling weights. Energy distances retain original window vectors, use all within/cross-sample pairs via cached block-distance means, and report the square root of the energy V-statistic. Nonzero within-block off-observation distances are retained.

## Reproducible results

| Representation | Contrast | L64 | L128 primary | L256 |
|---|---|---:|---:|---:|
| Z | N1-A2 | 0.03267165 | 0.03238800 | 0.03282653 |
| Z | A2-N2 | 0.05184610 | 0.05214824 | 0.05300603 |
| Z | N1-N2 | 0.06499340 | 0.06647815 | 0.06585806 |
| S | N1-A2 | 0.01209784 | 0.01205845 | 0.01209198 |
| S | A2-N2 | 0.02707765 | 0.02721030 | 0.02729633 |
| S | N1-N2 | 0.03347057 | 0.03417910 | 0.03387137 |
| S_tilde | N1-A2 | 0.01283639 | 0.01317494 | 0.01308396 |
| S_tilde | A2-N2 | 0.02971258 | 0.02978772 | 0.03006456 |
| S_tilde | N1-N2 | 0.03596998 | 0.03719445 | 0.03679853 |
| g | N1-A2 | 0.00525472 | 0.00587462 | 0.00552678 |
| g | A2-N2 | 0.01567453 | 0.01566921 | 0.01584628 |
| g | N1-N2 | 0.01878531 | 0.01969583 | 0.01953741 |

The full-precision point estimates, resampling standard deviations and q025/q975 for all 36 combinations are in r3_results.json. Keep L128 as primary; both other settings remain sensitivity results.

## Evidence supported

Across all four representations and all three tested settings, D(N1,N2) > D(A2,N2) > D(N1,A2). The same-actor cross-period comparison is the largest. The ordering persists under these prespecified changes to block length, sampled timestamps, and sample size.

Apart from g for N1–A2, sensitivity point estimates differ from primary by at most 4.63%. For g, N1–A2, L64 is 10.55% lower and L256 is 5.92% lower. These are descriptive comparisons, not significance tests.

## Limits and unsupported conclusions

- This does not establish actor-specific separation, semantic coherence, anomaly-detection performance, or superiority of one representation level.
- Raw distance magnitudes across levels have different coordinate scales/dimensions and cannot establish hierarchical improvement.
- Temporal/contextual variation is a possible explanation for the ordering, not an identified cause. Actor and recording period are not independently varied here.
- Resampling quantiles are conditional descriptive variability; they are not calibrated population confidence intervals or p-values.
- Blocks may retain dependence beyond their boundaries. Conditions are resampled independently; cross-period covariance and model-training uncertainty are not modeled.
- Block length, selected time ranges, and sample size change together. The sensitivity runs do not isolate a causal effect of block length and are not independent replications.
- Evidence concerns this CU checkpoint and the sampled periods; training convergence and generalization to other homes/residents are not established here.

## Numerical audit

36 point estimates and all 36,000 resampling values were independently recomputed from the saved distance tables using NumPy mean/einsum expressions, without calling the HDEG energy helper. Standard deviations and quantiles matched. Maximum absolute difference in resampling values: 2.533e-16. Absolute tolerance 1e-12; relative tolerance 1e-10.

Selected indices matched saved plans/hashes; seeded weights reproduced exactly; distance tables were finite, nonnegative, correctly shaped and symmetric for within-condition comparisons. Preflight and parent-report/plan hashes matched the archive. Two FAILED primary attempts were excluded. Their errors document configuration mismatches, not completed results.

The audit starts at saved block-distance tables. It does not independently verify original representation extraction or recompute distances from tensors. Numerical consistency is not confirmation of the scientific hypotheses.

## Level 1 status and next action

| Component | Evidence status |
|---|---|
| R3 primary + sensitivities | Executed; saved-summary numerical audit passed |
| R1 | No completed output supplied in this archive |
| R2 | No completed output supplied in this archive |
| Qualitative evidence and overall claim synthesis | Outstanding |

The inspected run_v1b.py --execute invokes R1, R2 and the older integrated R3 together and rewrites preflight.json. Its integrated R3 is not the standalone frozen-block analysis verified here. Preserve the completed R3 directories and current preflight; do not use the combined command as the next execution route.

Next implementation: a standalone R1 runner using the existing configuration imports, shared paths and checkpoint checks, with separate output. First review its computational work and the interpretation of within-chunk distance reservoirs. The current R1 pools local within-chunk distances across splits; it is not global geometry or a direct semantic-coherence test. R2 then follows separately; its current pooled Spearman correlation can reflect between-condition differences. Neither needs to rerun R3.

## Suggested manuscript wording (provisional R3 paragraph)

For the evaluated CU checkpoint, representation-distribution distances consistently followed D(N1,N2) > D(A2,N2) > D(N1,A2) across all four hierarchy levels. This ordering persisted in the prespecified 64- and 256-window sensitivity settings around the 128-window primary analysis, each using 32 sampled temporal blocks per condition. The findings characterize differences between the sampled recording periods but do not support interpreting the geometry as actor-specific separation. Reported resampling quantiles describe variability conditional on the selected blocks; they are not population confidence intervals. Sensitivity comparisons jointly vary block length, selected windows and sample size.
