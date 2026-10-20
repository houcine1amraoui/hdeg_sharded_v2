# HDEG — Prospective Level 2 operational protocol, v1

20 October 2026. Protocol ID: `hdeg_l2_learning_protocol_v1`.

## Status and purpose

This document operationalizes the manuscript's Level 2 Learning Process propositions before choosing another training experiment. It specifies evaluation and interpretation rules. It is **not a retrospective preregistration**, does not change the original objective or architecture, and does not declare existing experiments confirmatory.

The rules below are proposed prospective additions to the manuscript, informed by the completed exploratory work. The next experiment must reference this document and freeze its experiment-specific fields before execution. No learning rate, new architecture, training duration, or new training run is selected here. Data eligibility is unresolved; consequently no new experiment is yet ready for confirmation.

Existing evidence retains its current boundary: the evaluated HBF configurations underperform persistence, and the separate linear probe provides modest, uneven exploratory predictability. The checkpoint is `4b036348b014652bb84e0f5b44a2b46374bae00213f4ad07b1d4746ff862157b`. New protocol criteria must not be used to relabel these past runs as prospectively validated.

## 1. Scientific questions and permitted conclusions

| Question | Operational evidence | Permitted scope |
|---|---|---|
| O: Does the specified learning procedure execute numerically and make useful optimization progress? | Finite losses/gradients/parameters, actual parameter updates, fixed-panel error trajectories, representation variation, and a declared terminal endpoint. | Numerical behavior and progress of the recorded procedure under its finite budget. This is not mathematical convergence or stability across seeds. |
| T: Does the trained HBF provide temporal predictive benefit? | Paired per-level forecasting errors against persistence and a fitting-only mean-increment baseline; blockwise consistency and concentration; conditional resampling sensitivity. | Predictive utility for the selected windows and checkpoint, with explicit temporal coverage. |
| G: Does that benefit extend beyond optimization and selection data? | The same T analysis on eligible confirmation observations, unseen by the entire fitted pipeline and unused for model/protocol selection. | Transfer to that held-out temporal panel under its documented normal-operation definition. Not cross-home, cross-resident, or population-wide generalization. |

“Stable and meaningful representations” is not reduced to a loss threshold. This protocol measures numerical execution, predictive utility, and variation. Semantic meaningfulness, hierarchy-specific complementarity, and anomaly discrimination remain separate unestablished propositions. The manuscript must retain this limitation even if O, T, and G later receive positive bounded evidence.

## 2. Data roles and eligibility

Use three explicit roles; do not infer a role merely from a directory named `train`, `val`, or `test`.

| Role | Allowed use | Status of current data |
|---|---|---|
| Fit | Fit preprocessing, encoder, HBF and baseline statistics. | Existing training data have this role. |
| Development | Inspect diagnostics, choose hyperparameters/design, debug and select checkpoints using a declared rule. | Existing repeatedly inspected validation and train probes are development evidence. |
| Confirmation | Evaluate one frozen candidate/protocol once, after all fitting and selection decisions. | No eligible panel has yet been demonstrated. |

Confirmation eligibility requires a provenance ledger establishing all of the following:

1. No use in encoder, HBF, or preprocessing fitting, including imputation/scaling statistics. Test-time transformations may use only already-fitted state or a predeclared causal operation.
2. No direct or indirect use of this panel's outcomes for model, rate, checkpoint, metric, sampling, or stopping-rule selection. An unreported portion of a repeatedly inspected period is not automatically pristine.
3. No shared raw observations between confirmation pairs and fit/development pairs. Check the union of input and target timestamp support, not just distinct sample indices. Also account for preprocessing dependencies and boundary interpolation.
4. A documented reason to treat the panel as normal-operation data. Actor/period labels alone do not prove event-level normality. Where labels are unavailable, describe the assumption explicitly and restrict the conclusion accordingly.
5. The panel is not already reserved for another scientific question without an explicit revision of the study plan. Do not silently repurpose Actor 2 or the later Actor 1 period.

Disjoint raw spans do not guarantee statistical independence; record time gaps and remaining temporal dependence. Re-splitting encoder training data does not make it unseen to that encoder. A new complete-pipeline fit excluding a panel may fix training exposure but does not erase previous outcome-guided selection.

If eligibility cannot be established, continue only with an explicitly exploratory/development interpretation. Record G as `NOT_ASSESSED`; do not manufacture an independent holdout or infer confirmation from a new filename. A negative or inconclusive Level 2 outcome remains scientifically reportable.

## 3. Fixed evaluation budget and sampling

Retain the bounded evaluation scale: **2,048 fit-panel pairs and 2,048 comparison-panel pairs**, each organized as **32 chronological blocks of 64 consecutive window pairs**. The comparison panel is development or confirmation according to Section 2. Do not pool the panels for a scientific endpoint.

For already-used development analyses, retain the existing frozen selection file. For a newly eligible panel, select as follows before representation extraction or forecast scoring:

- Form all valid starts of 64 consecutive pairs whose entire raw support lies within one eligible continuous recording segment and one data role.
- Order valid starts chronologically and divide them into 32 approximately equal-count chronological strata. Using NumPy `Generator(PCG64(seed))`, choose one valid start per stratum: seed 42 for a newly selected fit panel, 43 for a newly selected comparison panel. Persist the selected indices, timestamps, strata and library versions; saved selections, not regeneration assumptions, are authoritative.
- Prevent raw-support overlap between selected blocks: process strata chronologically, exclude starts overlapping an already selected block, and sample from the remaining candidates. Record exclusions. If any stratum has no eligible start, declare the design infeasible; do not silently shrink the panel, duplicate blocks, or change the window length.
- Select solely from eligibility and timestamps, never from increment size, HBF error, activity, or visible benefit. Preserve all selected blocks in the primary results.

The current forecast definition remains a 30-observation input window and a target shifted by one observation, at the established one-second sampling interval. Consecutive input/target windows share 29 observations. Any changed horizon or sampling interval creates a separate protocol version.

This fixed-panel estimand weights the 32 selected blocks equally. It is not automatically the time-weighted risk of the full home recording. Within a block, overlapping windows are retained; they are not treated as independent experimental units.

## 4. Prediction contract and baselines

For each level r in {Z, S, S̃, g}, flatten its target representation into d_r coordinates. Under a fixed evaluated encoder, let a_i be the current representation, y_i the next-window representation, and h_i the HBF forecast. All comparators use identical pairs, encoder state, device order, coordinates, target values, and evaluation mode.

- **HBF:** h_i from the declared model and checkpoint.
- **Persistence P:** a_i.
- **Mean-increment M:** a_i + μ_r, where μ_r is the coordinatewise mean of y_i − a_i on the 2,048-pair fit panel only.

Fit μ independently for each evaluated frozen encoder; never estimate it from the comparison panel. Report the fitting sample size and mean-increment vector hash. A train-target mean baseline may remain a secondary diagnostic, but does not replace either primary comparator. The existing ridge predictor is a separate exploratory architecture, not HBF evidence.

Use float64 accumulation for squared errors and summaries. No test-fitted normalization, oracle centering, clipping, or event-dependent gate enters the primary prediction. Existing model preprocessing remains unchanged. Different checkpoint encoders create different latent targets: raw MSE decreases across such checkpoints do not alone establish learning improvement.

## 5. Primary endpoints and operational decisions

For method m, window i and level r:

`e[m,i,r] = mean_coordinate((prediction[m,i,r] − y[i,r])²)`.

For block b, `E[m,b,r]` is the mean of its 64 window errors. For baseline q in {P, M}:

- Paired block difference: `D[q,b,r] = E[HBF,b,r] − E[q,b,r]` (negative favors HBF).
- Primary effect: `Δ[q,r] = mean_block(D[q,b,r])`.
- Relative skill: `S[q,r] = 1 − mean_block(E[HBF,b,r]) / mean_block(E[q,b,r])` when the denominator is strictly positive.
- Temporal consistency: number of blocks with D < 0, exact ties, and D > 0; median D; full block table.

Always show absolute HBF and baseline MSE alongside skill. Do not average the four latent-level errors into a single pass criterion. If a baseline MSE is zero, report relative skill as undefined and use absolute differences; a nonnegative squared-error forecaster cannot strictly outperform zero error. Do not add an epsilon to create an apparent skill value.

The following are **new prospective descriptive rules**, not significance tests or claims that a practical minimum effect has been established:

| Outcome per level | Rule |
|---|---|
| `SUPPORTED_ON_PANEL` | Δ < 0 against both baselines; at least 17/32 blocks improve against each; the upper endpoint of each specified conditional resampling range is < 0 under both resampling designs in Section 6; integrity and numerical-resolution checks do not leave the direction unresolved. |
| `ADVERSE_ON_PANEL` | Aggregate HBF error is strictly greater than persistence error (Δ[P,r] > 0). Still report M, all block results, and uncertainty; this is an observed negative result, not proof of population harm. |
| `MIXED_OR_UNRESOLVED` | All other valid outcomes, including equality, mixed baseline comparisons, gain concentrated in a minority, ranges crossing zero, or unresolved numerical precision. |
| `NOT_ASSESSED` / `INVALID` | Missing eligible evidence / violated pairing, provenance, finite-output, or other required execution contract. These are not negative scientific measurements. |

The majority requirement supplies a minimal operational meaning for “consistent” over the selected blocks; it does not mean every period benefits. The zero-effect boundary measures direction, not practical importance. No scientifically justified practical-effect margin is currently available; report effect magnitude without calling a small effect practically important.

A positive **all-level temporal result** requires `SUPPORTED_ON_PANEL` at all four levels. Partial success is reported level by level, without choosing the best level after inspection. This conjunction and the ranges do not provide familywise statistical error control; no p-value or significance claim is made. On a development panel, even an all-level positive result remains exploratory. On an eligible confirmation panel it supports only the bounded G claim in Section 1.

## 6. Temporal uncertainty and concentration

Use **1,000 paired resampling replicates**, with seed 20261020, and save the actual resampling index arrays. Use the same draws for every method and representation level.

Primary conditional resampling: sample 32 of the selected blocks with replacement per replicate. Recompute Δ and skill from the resampled block errors; compute skill from resampled aggregate errors, not the mean of blockwise skills. Report the empirical 2.5th and 97.5th percentiles using NumPy's linear quantile convention.

Prespecified dependence sensitivity: order the 32 blocks by time, group adjacent selected blocks into 16 pairs, and sample 16 pairs with replacement per replicate, retaining both members. Again use 1,000 replicates and paired draws across methods/levels, with a separately initialized generator at seed 20261021. Report this as a sensitivity to coarser temporal grouping; time gaps between selected blocks may differ.

Both procedures produce **conditional block-resampling ranges**, not population confidence intervals. They condition on the selected blocks, fitted model, baselines, data preparation, and chosen protocol. They do not include training-seed uncertainty, baseline-estimation uncertainty, unseen-regime variability, or arbitrary long-range dependence. Pair grouping does not prove independence. No IID window bootstrap, IID window-level test, or nominal 95% significance claim is permitted.

For every level, report the contribution of each block to baseline error and to net HBF improvement, with signed values. Also report fixed leave-one-block-out aggregate differences for all 32 blocks as a concentration diagnostic. Do not remove harmful blocks, redefine the primary panel, or use true future increment magnitude to gate a deployed prediction. Any plots stratified by observed target increment are descriptive only.

## 7. Representation variation and numerical resolution

For current representations, targets, true increments u = y − a, predicted corrections c = h − a, and residuals v = h − y, save coordinate means and centered second moments per block and panel. Report mean energy, centered variance, increment RMS, correction RMS, and correction–increment covariance. This permits the identity:

`MSE(HBF) = mean(c²) + mean(u²) − 2 mean(c*u)`.

Also report the residual mean-offset term and centered residual variance whose sum is MSE. These are descriptive decompositions; comparison-panel means must not become fitted corrections. Save small coordinate-moment arrays rather than all activations when disk is limited.

Record input-window variation as context. Small latent variation alone is neither semantic stability nor collapse; low input variability may also matter. Comparisons across changing encoders must flag changes in target scale and geometry. Variation guards against interpreting a shrinking target space as forecast success, but does not certify behavioral meaning.

Numerical checks must cover finite values, decomposition agreement, error aggregation, and baseline reproduction. Before a future run, freeze dtype-specific tolerances using synthetic arithmetic and development-only checks; do not widen them to accept confirmation results.

As a bounded inference-repeat diagnostic, repeat extraction/forecasting for the first two selected windows of each block: 64 pairs per panel, with unchanged weights and settings. Report maximum and RMS representation/forecast discrepancies and changes in paired error differences. This is 128 extra pair evaluations across both panels, with no optimization. It is an empirical repeatability check, not a bound for every window. If observed numerical variability can change an asserted improvement's direction, mark it unresolved and investigate before interpreting it. Unknown numerical resolution cannot be dismissed merely because a scalar error is positive or a solve residual is small.

## 8. Optimization evidence for any later experiment

These are recording requirements for a future declared procedure, not instructions to train now.

Before execution, freeze: model/encoder update scope, initialization/checkpoint hashes, optimizer and all hyperparameters, minibatch order/seed, number of updates U, batch size, evaluation schedule, terminal endpoint, and selection rule. Default scientific endpoint is the terminal step U; an intermediate best result cannot replace it after inspection. A numerical failure stops execution and is reported. Budget exhaustion is not convergence.

Use a sparse fixed schedule `{0, 1, ceil(U/4), ceil(U/2), ceil(3U/4), U}`, deduplicated, for fit/development monitoring. Confirmation is evaluated only after the candidate is frozen, not at every training checkpoint. This schedule's inference cost must be included in the future budget before U is selected.

Log every update's total and per-level losses and finite-status; record gradient and actual parameter-update summaries on the first eight updates (or all if U < 8) and at scheduled endpoints. Separate encoder, shared backbone, head weights, and head biases. When target encoders remain live in autograd, record that fact explicitly. Zero upstream gradient at a deliberately zero-initialized head's first step is not automatically a wiring failure.

Use these distinct O outcomes:

- **Numerically executed:** declared updates completed with finite required quantities and traceable state. Does not establish useful learning.
- **Bounded optimization progress:** for a fixed encoder, terminal fixed fit-panel per-level errors improve over initialization at all four levels; report development behavior and baseline skill separately. Otherwise report mixed/no progress. Do not require every minibatch loss to decrease.
- **Changing encoder:** record normalized skill, target variation, and loss trajectories, but do not declare progress from decreasing raw latent MSE alone. A positive utility claim still requires T. Cross-checkpoint representation semantics/stability remain outside this rule.

No O outcome in one bounded run warrants “converged,” “robust across seeds,” or a causal claim that the weighted objective uniquely produces meaningful representations. Frozen-encoder head optimization tests a narrower procedure than the manuscript's joint end-to-end learning claim.

## 9. Evidence record and configuration contract

Preserve the established configuration-loading style: `configs/config.yaml`, the existing `yaml.safe_load` pattern, `--project_root_dir`, and the shared processed-path helper. This protocol introduces no replacement configuration system. A future standalone runner should read only its own explicitly named settings and record resolved values; missing mandatory settings must not silently inherit unrelated pilot settings.

Retain a compact record containing protocol/version hash, configuration snapshot/hash, source hashes, model checkpoint hash, data-role eligibility, selection file, baseline statistics, final per-window errors, block/coordinate moments, resampling indices, numerical checks, timing, and separate execution/scientific statuses. Do not load artifacts with mismatched checkpoint identities. Model selection and inference artifacts must trace to the same candidate.

No full duplicate artifact tree or mandatory checkpoint archive is introduced here. Respect disk limits: retain compact summaries and hashes, and never describe a replaced checkpoint as recoverable. If weights are later removed, explicitly record the resulting reproducibility limitation. Do not overwrite completed evidence reports merely to store a new experiment's result.

Use separate fields conceptually equivalent to `execution_status`, `O_status`, `T_status_by_level`, `G_eligibility`, `G_status_by_level`, and `claim_scope`. `COMPLETE` must never mean scientific acceptance. A single Level 2 pass boolean would hide unresolved semantics and the differences between optimization, prediction, and generalization.

## 10. What is fixed now and what remains open

Fixed prospectively in this document: the three questions; development/confirmation distinction; unchanged forecast horizon; 32 × 64 pairs per panel; paired HBF/P/M comparisons; per-level endpoints and descriptive decisions; 1,000 paired resamples with a coarser grouping sensitivity; variation/moment reporting; and evidence interpretation limits.

Required before any next experiment: eligible data roles and timestamp support, candidate update scope and initialization, U and optimizer settings, exact source/configuration/selection hashes, numerical tolerances, computational feasibility, and the relationship of current Level 1 evidence to the candidate encoder. These fields are intentionally unresolved because choosing training is the next decision, not this task.

**Next focused step: a read-only data-use and provenance audit.** Use existing split manifests, preprocessing-fit records, selection plans, run reports and timestamps to determine whether any normal-operation confirmation panel is eligible. This audit needs no model training. If no such panel can be demonstrated, record that limitation and decide explicitly between further exploratory development and obtaining suitable confirmation data. Do not choose a learning rate or another training run first.

## 11. Manuscript integration boundary

Proposed protocol wording, for later insertion:

> Level 2 separates numerical optimization behavior, temporal forecasting utility, and transfer beyond optimization and selection data. Forecasting utility is evaluated per representation level against persistence and a fitting-only mean-increment baseline on identical temporally grouped window pairs. Aggregate effects are interpreted jointly with blockwise consistency, representation variation, and conditional block-resampling sensitivity. These analyses do not by themselves establish semantic meaningfulness, population-level generalization, or anomaly discrimination. Development diagnostics and eligible confirmation evidence are reported separately, and adverse or inconclusive results are retained as validation outcomes.

The actual manuscript and implementation specification remain unchanged. Their successful-validation language should be qualified when the protocol is integrated. Levels 1, 3 and 4 are outside this implementation step.

## Sources

This protocol is a prospective design based on the already-reviewed sources, not an additional empirical result:

- `HDEG_Level2_Manuscript_Claim_Map.md`: exact manuscript propositions, references and qualification needs.
- `HDEG_Level2_Evidence_Boundary.md`: eight completed diagnostic reports and their limitations.
- `cas-sc-template.tex`, Section 5.3 “Learning Process,” lines 862–870; SHA256 `ad800c06db49c42d561a92a61642576986e327fcde998770f0c124e100b80d41`.
- `implementation specification(3).tex`, SVE and Scientific Claim Synthesis contracts; SHA256 `b1c6afd2f42076f725d87666ace382b4f05189d832914471d448c92fa9556804`.
