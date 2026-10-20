# HDEG — Current Level 2 evidence boundary

Consolidated 20 October 2026. Scope: the completed forecasting diagnostics and follow-up probes. This note freezes their interpretation; it does not amend the manuscript, implementation specification, production model, or other validation levels.

## Status and governing conclusion

**The completed runs establish an executable, auditable diagnostic pipeline, but do not establish successful Level 2 forecasting validation for the evaluated HBF.** `COMPLETE` describes execution, not acceptance of a scientific claim. The current forecasting claim remains unsupported by the evaluated HBF configurations. This is a checkpoint- and protocol-specific conclusion, not a rejection of the whole HDEG concept or a proof of impossibility.

All eight reports identify the same original one-epoch checkpoint:

`4b036348b014652bb84e0f5b44a2b46374bae00213f4ad07b1d4746ff862157b`

None of the exploratory HBF variants, calibrated rates, or linear probes has been promoted to the production model. Their evidence must remain distinct from the original framework's results. This note makes no new assertion that Level 1 or Levels 3–4 are complete.

## Original and exploratory HBF evidence

The original diagnostic used 2,048 selected training and 2,048 selected validation pairs, with 32 blocks of 64 windows per split. The frozen-encoder HBF-only and persistence-initialized residual-HBF pilots reused those selections and each ran for a predetermined 640 updates. Final step 640, not an intermediate best step, is the reported endpoint.

Validation MSE divided by persistence MSE; lower is better, and 1 is parity:

| Level | Original HBF | HBF-only pilot, step 640 | Residual-HBF pilot, step 640 |
|---|---:|---:|---:|
| Z | 573.12× | 548.22× | 1.668× |
| S | 645.13× | 972.91× | 1.693× |
| S̃ | 547.72× | 834.77× | 1.752× |
| g | 1170.09× | 1471.07× | 1.862× |

All three configurations underperformed persistence at all four levels. The original diagnostic and the two pilots provide no positive forecasting-skill result beyond persistence. The residual formulation exactly reproduced persistence at initialization, but its trained correction worsened the fixed final endpoint.

The gradient audit found finite, connected total gradients in the audited batches. This rules out a completely disconnected training graph in those measurements; it does not prove that optimization is effective or that historical training converged. Frozen-encoder optimization also failed to recover forecasting skill under its bounded budget, so a moving encoder is not necessary for the observed disadvantage. This is not unique causal attribution.

Read-only decomposition showed large validation mean-offset contributions for the original and HBF-only forecasters. Descriptively removing those offsets still left centered error above persistence. The train-target mean baseline also outperformed both forecasters at S, S̃, and g on validation. Split-specific centering is an oracle descriptive calculation, not a deployable correction.

## Optimization-scale findings

The train-only eight-update replay measured actual parameter displacements and head contributions. Its first update worsened its own training minibatch at every level; weight and bias contributions reinforced one another. Shared head inputs varied, but their energy was dominated by the mean component. Epsilon was relevant to the first Adam update. These observations support an initial update-scale concern without establishing a single cause of all forecasting failures.

The fixed calibration tested rates 1e-4, 1e-5, 1e-6, 1e-7 and 1e-8, each for eight updates from identical initialization with fresh Adam state. Lower rates reduced overshoot and could improve the first update's own minibatch. No rate beat persistence on the fixed training probe at the evaluated post-update endpoints. Approaching persistence through tiny corrections is not evidence of useful learned dynamics.

The calibration probe had distinct rows but all 256 windows shared raw observations with update windows. It is a local training diagnostic, not an independent holdout. No rate was selected for a production configuration or longer run.

Historical replay agreement failed the predeclared tolerance in both the instrumented replay and calibration control. Maximum recorded relative loss discrepancies were approximately 0.95% and 0.255%, respectively. Preserve these flags; later instrumented measurements are not exact reconstructions of the original pilot trajectory. Within-run calibration comparisons use the same cached inputs and identical starts. Matching software labels or a small discrepancy does not resolve its cause.

## Bounded linear increment evidence

A separate ridge-linear diagnostic used the full current hierarchy to predict representation increments, bypassing HBF. The first 16 selected training blocks (1,024 windows) fitted coordinate means, per-level scales, mean increments and the ridge predictor. The last 16 blocks (1,024 windows) evaluated them. One penalty, 0.01, was fixed; no tuning was performed. The fit/evaluation raw observation spans do not overlap and are separated by about 10.29 hours.

Aggregate evaluation MSE reductions:

| Level | Over persistence | Over fit-only mean increment | Blocks improved over persistence |
|---|---:|---:|---:|
| Z | 0.409% | 0.406% | 5/16 |
| S | 0.876% | 0.875% | 5/16 |
| S̃ | 0.809% | 0.808% | 5/16 |
| g | 0.955% | 0.952% | 5/16 |

This is modest, uneven evidence of accessible linear predictive information under the particular frozen-encoder specification. Improvement over the mean-increment baseline distinguishes it from simply predicting average drift. It does not validate HBF, demonstrate a hierarchy-specific advantage over alternative representations, or establish anomaly-detection performance.

The encoder originally trained on both groups, and these windows appeared in earlier exploratory diagnostics. Temporal separation protects the probe's fitting/evaluation split from shared raw observations; it does not turn this into independent end-to-end validation. The small linear-system residual supports numerical solution accuracy, not statistical significance or robustness to encoder-inference variation.

## Block-level boundary

At all four levels, blocks 17, 22, 26, 27 and 30 are both the five highest-increment-energy evaluation blocks and the only five where the linear predictor beats persistence. The remaining eleven worsen, and the median block error difference is unfavorable.

Blocks 22 and 26 contribute approximately 83.2% of persistence error for Z and 97.3–98.0% for S, S̃ and g. They account for about 98.7–100.5% of the net error reduction. A contribution above 100% means the other blocks collectively offset some of the gain. Excluding both descriptively leaves skill of −0.012%, +0.295%, +0.209% and +0.635% for Z, S, S̃ and g. These post-hoc exclusions are sensitivity observations, not replacement primary results.

In the worsening blocks, median representation-increment RMS is approximately 1–2e-9. The fitting-set mean increment has RMS approximately 1.9–4.5e-8. This fixed correction worsens persistence in every one of those blocks, with added error close to its own squared magnitude. Constant offsets therefore demonstrably penalize near-persistent representations in this baseline.

For the linear predictor, the saved per-window scalar errors establish harmful corrections in quieter blocks but do not identify their local mean/variance/covariance decomposition. No block-specific prediction vectors or coordinate moments were saved. Its quiet-block harm cannot be attributed exclusively to constant offsets. Do not transfer the residual-HBF correction-variance findings to this different linear predictor.

“Larger changes” and “quieter” here refer to learned representation increments. No device-level event or anomaly labels were examined. Extremely small increments may include numerical effects; they are not automatically meaningful physical changes.

## Claim boundaries

Supported within the stated scope:

- The audited HBF configurations underperform persistence on the fixed paired windows.
- Initial residual-HBF update scale is problematic under the measured settings; lowering the rate reduces damage without establishing probe benefit.
- One fixed linear increment probe gives small aggregate gains over both baselines on later training blocks, concentrated in larger-change blocks.
- Constant mean-increment offsets harm near-persistent blocks; the linear predictor also worsens those blocks.

Not established:

- Positive HBF forecasting skill or successful completion of the intended Level 2 forecasting claim.
- Uniform temporal benefit, statistically significant improvement, independent end-to-end generalization, or robustness across checkpoints/seeds/homes.
- Complete representation collapse, absence of all predictability, one uniquely identified failure cause, or superiority attributable to the hierarchy itself.
- Anomaly-detection accuracy, threshold suitability, or a benefit for MBAI.

No IID tests or confidence claims are warranted from overlapping windows. Validation has repeatedly informed exploration and is not a pristine confirmatory set. Any new architecture, gating rule, normalization, rate, horizon, or objective must be a separately versioned experiment with fitting-only choices and an explicit confirmation protocol. Future increments cannot serve as an inference-time gating signal.

## Candidate manuscript wording

The following is a standalone evidence paragraph for later integration, not an edit to the current manuscript:

> In bounded diagnostics of the evaluated one-epoch checkpoint, the original hierarchical forecaster and two frozen-encoder optimization variants underperformed persistence at all four representation levels. Instrumented training revealed substantial initial correction-scale errors; reducing the learning rate mitigated these errors but did not produce improvement on the training probe. A separate fixed ridge-linear increment predictor achieved aggregate MSE reductions of 0.41–0.96% over persistence on later, non-overlapping training blocks, with similar gains over a fitting-set mean-increment baseline. However, only five of sixteen evaluation blocks improved at each level, and gains were concentrated in blocks with larger representation increments. These exploratory results indicate limited, uneven linear predictability in the frozen hierarchy, while providing no positive validation of the evaluated HBF forecasting mechanism. Because the encoder had trained on both temporal groups, the linear-probe result does not constitute independent validation of the complete framework.

## Decision at this checkpoint

Consolidate and retain the negative HBF findings and qualified linear-probe result. Do not silently replace HBF with persistence or the linear predictor, promote a calibrated rate, add an activity gate, or mark the forecasting claim as passed. Further model development remains possible, but constitutes new experimental work. Before presenting the paper's claims as supported, map this boundary to the manuscript's exact Level 2 claims and identify which must be qualified, revised, or tested anew.

## Source report ledger

These reports were already reviewed in the session. This consolidation checks their completed status and common checkpoint and records SHA256 values for traceability. It does not rerun inference or reconstruct unsaved activations. Supporting block attribution is recomputed from the linear run's saved per-window errors. Hashes identify artifacts; they do not certify scientific validity.

- **Original forecasting diagnostic** — run `20261018T190648Z_a76a55fd`, protocol `v2_forecast_32x64_v1`; report `diagnostic_report.json`. SHA256: `5370a1c11bbf0e123ba88b9f25cc78b860412c73bf1824a04557cdc025af358e`.
- **Gradient audit** — run `20261019T081240Z_3fae1dd0`, protocol `v2_gradient_4x32_v1`; report `gradient_report.json`. SHA256: `75802bc842fc85bf184b5a4ad8678c16883383ad83f46116cb721f18b8dff266`.
- **Frozen-encoder HBF pilot** — run `20261019T130203Z_fa27faeb`, protocol `v2_hbf_only_640_v1`; report `pilot_report.json`. SHA256: `ec13b927ff4118239cc838438e997f4f9c1928ff4487f2c11562ad9421d4dca7`.
- **Residual decomposition** — run `20261019T144840Z_da8bc7cb`, protocol `v2_residual_decomposition_v1`; report `residual_report.json`. SHA256: `88003c2463b84329865f0d51071539c55120c2cf73c1bcd6642ffca22e15c982`.
- **Residual-HBF pilot** — run `20261019T180441Z_2b3ef3d3`, protocol `v2_residual_hbf_640_v1`; report `residual_hbf_report.json`. SHA256: `981c3ff38081d0e0d3de25807e46e136780bb770c50f49fd697d4acacbfa77e7`.
- **Instrumented replay** — run `20261019T182137Z_70ec8d75`, protocol `v2_train_only_8_update_replay_v1`; report `optimization_replay_report.json`. SHA256: `b84884ff072dd4374063bd87b23d4c2feae42bca1f3eadd2d51564664fef9312`.
- **Learning-rate calibration** — run `20261020T055224Z_3e868b3e`, protocol `v2_train_only_lr_grid_5x8_v1`; report `lr_calibration_report.json`. SHA256: `7438d2a2599bcaed8c0df1bcee25915cb45976199d4191127ede6b80e210b18d`.
- **Linear increment probe** — run `20261020T075653Z_9cafbce3`, protocol `v2_train_linear_increment_16_16_v1`; report `linear_predictability_report.json`. SHA256: `2902643cf24ddecc0cc5e7d90973694fb39a498307a18b1b9d8d226c79c6ae8a`.
