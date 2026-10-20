# HDEG — Level 2 manuscript claim-to-evidence map

20 October 2026. Read-only synthesis of the current manuscript and the consolidated Level 2 evidence boundary. No manuscript, configuration, model, or validation script is changed. No additional experiment is selected or run.

## Governing finding

**Level 2 is not scientifically established for the evaluated checkpoint.** The manuscript defines Level 2 as validation of the learning process: stable optimization, consistent temporal modeling, and generalization beyond optimization data. The completed diagnostics provide useful negative and exploratory evidence, but do not jointly establish these propositions.

This does not mean every proposition was directly falsified. The clearest adverse observation is that the evaluated HBF configurations underperform persistence. Historical end-to-end optimization stability, semantic meaningfulness, and broad generalization remain insufficiently assessed. Finite gradients and completed execution are narrower technical findings.

The manuscript does **not** prescribe persistence superiority, a numerical acceptance threshold, a sampling budget, or an uncertainty procedure for Level 2. Persistence was an added diagnostic comparator, not a pre-existing manuscript pass/fail rule. It is especially informative for the evaluated 30-observation windows shifted by one observation. Failure against this baseline undermines a positive forecasting interpretation under this protocol; it is not a universal impossibility result or, by itself, a test of all Level 2 propositions.

## Source scope and traceability

Primary text: `cas-sc-template.tex` extracted from `manuscript v2.zip`, Section 5.3, “Learning Process,” source lines 862–870. Exact quotations below are from this source. Its “Learning Validation Results” subsection at line 1007 is currently empty: the claims under review are validation propositions and methodological assertions, not an already-written Level 2 results narrative.

Supporting sources:

- Section 3.3, “Hierarchical Behavioral Forecasting”; Section 3.5, “Model Optimization”; Section 4.4, “Experimental Protocol”; and Section 5.2, “Representation Quality.”
- `implementation specification(3).tex`, Scientific Validation Engine (SVE), especially source lines 3028–3032, 3050–3058, and 3181–3201; Scientific Claim Synthesis contract at line 3582.
- `HDEG_Level2_Evidence_Boundary.md`, including its eight-report ledger and block-level interpretation. This map uses that consolidated evidence; it does not rerun model inference or reopen the original training trajectory.

All eight Level 2 reports in that boundary refer to the one-epoch checkpoint:

`4b036348b014652bb84e0f5b44a2b46374bae00213f4ad07b1d4746ff862157b`

This map does not certify that every manuscript Level 1 figure/result was regenerated from this checkpoint. Historical and current Level 1 evidence must retain their own checkpoint identities.

## Exact Level 2 propositions

Repeated wording in the four Level 2 paragraphs is grouped by proposition rather than counted as independent hypotheses. “Not established” means the available evidence does not justify the positive claim; it does not automatically mean the opposite has been proved.

| ID and source | Exact manuscript wording | Evidence relevant to this proposition | Current assessment and writing consequence |
|---|---|---|---|
| L2-A; line 864 | “effectively learns the normal behavioral dynamics of the smart-home ecosystem” | Original HBF and both bounded frozen-encoder pilots lose to persistence at all four levels on the selected train/validation pairs. The separate linear probe bypasses HBF. | **Not established; adverse forecasting evidence.** Retain this as the scientific question or objective, not a demonstrated outcome of the evaluated HBF. |
| L2-B; line 864 | “acquire stable and meaningful hierarchical behavioral representations rather than merely optimizing predictive performance” | Variation was measured in selected representations/shared inputs, but this does not establish semantic meaning or stability across training, seeds, or checkpoints. Level 1 itself restricts its claims to geometry. | **Not established by these diagnostics.** Define “stable” and “meaningful” operationally before proposing a confirming test. Nonzero variation is not semantic validation; near-persistence is not proof of collapse. |
| L2-C; lines 866 and 868 | “stable optimization behavior” | Audited total gradients are finite and connected. Residual-HBF replay exposes initial update-scale problems; smaller rates reduce damage. Bounded frozen-encoder runs do not recover skill. Historical replay agreement fails tolerance. | **Partial technical evidence, with adverse local optimization findings.** State gradient connectivity and measured update behavior separately. Neither proves stable/converged original end-to-end training. Do not generalize a residual-variant replay to the original training trajectory. |
| L2-D; line 868 | “consistently models the temporal evolution of normal behavioral patterns” | Evaluated HBF configurations underperform persistence. Linear increments improve aggregate MSE by 0.409–0.955% over persistence, but only 5/16 evaluation blocks improve at each level; gains concentrate in larger-increment blocks. | **Not established for HBF.** The separate probe gives limited, uneven accessible linear predictability, not consistent HBF temporal modeling. Do not redefine consistency post hoc as pooled mean improvement. |
| L2-E; line 866 | “behavioral knowledge that generalizes beyond the optimization data” | Original forecasting diagnostic includes 2,048 validation pairs, with unfavorable relative performance. Linear-probe evaluation is later training data seen by the encoder; the calibration probe also shares raw observations with update windows. | **No positive demonstration of end-to-end generalization.** The original validation comparison is relevant negative evidence. The linear probe is not independent generalization evidence for the full model. Subsequent exploration has repeatedly used validation. |
| L2-F; line 868 | “generalizes across previously unseen observations drawn from the normal operating conditions of the smart-home ecosystem” | Existing train-only probes do not satisfy unseen-to-the-complete-model status. The bounded validation diagnostic does not establish forecasting advantage or coverage across normal regimes. No new event annotation or cross-home confirmation was performed. | **Broader claim remains unestablished.** Any future test must identify what is unseen, to which fitted component, and under what verified normal-data definition. Representation-change magnitude alone is not a normal/anomalous event label. |
| L2-G; line 870 | “a stable, effective, and generalizable mechanism for learning the hierarchical behavioral dynamics of the smart-home ecosystem” | This is the joint conclusion of L2-A through L2-F. A successful diagnostic execution cannot substitute for the missing scientific support. | **Not established.** Report a bounded negative/qualified outcome rather than successful Level 2 validation. |
| L2-H; line 868 | “establishing its suitability as the foundation for the subsequent behavioral drift detection analyses” | No positive HBF validation, anomaly discrimination test, or MBAI benefit follows from the current Level 2 evidence. | **Foundation not established by these runs.** Do not advance a scientific pass to Levels 3–4 on this basis. This map makes no claim that those levels have failed an executed test. |

## What the evidence establishes more narrowly

The original diagnostic selected 32 blocks × 64 windows in each of train and validation: 4,096 paired windows in total. The two frozen-encoder pilots used the same selections and fixed 640-update endpoints. Validation MSE relative to persistence is:

| Representation | Original HBF | Frozen-encoder HBF pilot | Residual-HBF pilot |
|---|---:|---:|---:|
| Z | 573.12× | 548.22× | 1.668× |
| S | 645.13× | 972.91× | 1.693× |
| S̃ | 547.72× | 834.77× | 1.752× |
| g | 1170.09× | 1471.07× | 1.862× |

These are ratios on near-persistent latent targets, not errors in physical sensor units. Report absolute errors alongside them in eventual results. The ratios are useful relative comparisons, not semantic-meaning or anomaly-quality measures. Split-specific oracle centering still leaves error above persistence, so mean offsets alone do not explain away the disadvantage.

The ridge probe predicts increments from the full frozen current hierarchy. Its fit/evaluation split contains 1,024/1,024 training windows with non-overlapping raw spans and about 10.29 hours between groups. Its small aggregate gains over both persistence and fitting-set mean increments are a separate exploratory observation. Blocks 22 and 26 account for approximately 98.7–100.5% of its net error reduction; eleven quieter blocks worsen. This limits a claim of temporal consistency, even though the aggregate gain is positive.

No inferential significance or population confidence claim follows from these descriptive results. The report does not treat overlapping windows as independent observations. The block analysis was post hoc and must not become a retrospectively selected primary endpoint. The linear probe's quiet-block harm cannot be attributed exclusively to a constant offset because the necessary block-specific prediction moments were not saved.

## Adjacent manuscript assertions that need qualification

These passages are not additional Level 2 experiments. They explain why the evidence boundary matters beyond the short Learning Process subsection.

| Location | Exact wording | Implication of the evidence boundary |
|---|---|---|
| Section 3.3, line 544 | “thereby preserving temporal consistency throughout the behavioral hierarchy and providing complementary predictive information” | Joint multi-head prediction is a design fact; preservation and complementarity are empirical propositions. Current diagnostics do not establish either. Describe the design as intended to encourage these properties. Complementarity would require a separately defined comparison, not just four output heads. |
| Section 3.3, line 570 | “This shared representation captures the normal temporal evolution patterns” | The shared representation is computed, but capture of useful normal dynamics is not established by execution or gradients. Prefer design-intent wording until supported. |
| Section 3.5, line 718 | “learns hierarchical behavioral representations that are simultaneously semantically meaningful and temporally predictive” | Optimization of the stated loss is not a guarantee of semantics or useful prediction. Preserve the loss definition; qualify the asserted outcome. |
| Section 4.4, line 760 | “identify the model that best captures the forecasting dynamics” and “The resulting validated HDEG model” | A single one-epoch checkpoint and the present diagnostics do not document the claimed model-selection outcome or scientific validation. Distinguish the evaluated checkpoint from a model shown to satisfy the validation propositions. Do not infer best-model selection from a loaded checkpoint. |
| Section 5.3, line 864 | “the learned representation space established in the previous validation level” | Section 5.2, lines 779–786 explicitly says geometric structure does not establish semantics, forecasting reliability, or anomaly discrimination. Cross-reference those bounded observations rather than treating the broader representation proposition as established. |
| Section 5.3, line 870 | “This integrated interpretation establishes the scientific conclusion of the second validation level” | A conclusion may be negative or inconclusive. Make that possibility explicit rather than implying the synthesis must validate the proposition. |

The one-observation shift is explicit at manuscript line 219. Line 752 describes an “entire future behavioral context”; that wording should acknowledge that consecutive 30-observation windows share 29 observations in this evaluated protocol. It does not create a non-overlapping 30-step forecast. This explains the relevance of persistence without changing the forecast horizon after seeing results.

## Specification alignment

The SVE says validation criteria and assessment protocols are predefined (lines 3028–3032), preserves links from outcomes to propositions and artifacts (line 3024), and must leave the experimental record unchanged (line 3201). Its hierarchy-consistency test at line 3189 requires successful completion of preceding levels before higher validation levels are evaluated. The Scientific Claim Synthesis contract at line 3582 forbids unsupported inferences and generalizations beyond evidence.

The present manuscript Level 2 subsection gives scientific aims but lacks an executable definition of acceptance for those aims. This is a **protocol-specification gap**, not permission to invent a retrospective pass threshold. The completed pilots should remain explicitly exploratory. Technical completion, negative findings, inconclusive findings, and scientific support must be distinguished in any eventual evidence record.

Some specification language describes every intermediate validation state as successful validation. That wording needs a future clarification so adverse or inconclusive outcomes remain reportable without becoming positive “validated scientific propositions.” No specification change is made here. Under its current progression rule, this mapping does not authorize a successful Level 2 state or progression to higher-level scientific validation.

## Proposed wording for the current manuscript boundary

The following is proposed text only; it has not been inserted into the manuscript:

> Level 2 examines whether the hierarchical forecasting objective supports stable optimization, useful temporal prediction, and generalization beyond optimization data. These properties are treated as empirical propositions rather than consequences guaranteed by minimizing the training loss. For the evaluated one-epoch checkpoint, bounded comparisons found that HBF and two frozen-encoder optimization variants underperformed persistence at all four representation levels. Finite gradients established connectivity in the audited batches, while instrumented follow-up runs identified local update-scale concerns without establishing the stability or convergence of the original end-to-end training process. A separate linear increment probe yielded small aggregate gains on later training blocks, concentrated in five of sixteen blocks. Because the encoder had trained on both temporal groups and the probe bypassed HBF, this result does not establish HBF effectiveness or independent generalization of the complete framework. The current evidence therefore leaves the broader Level 2 proposition unestablished and does not establish the claimed learning foundation for subsequent behavioral drift validation.

The fuller evidence paragraph and numeric tables in the companion boundary remain applicable. Keep original-HBF evidence, variant-pilot evidence, and the independent *predictor architecture* of the linear probe separate; “separate predictor” must not be confused with an independent evaluation dataset.

## Decision before further experiments

**Do not select another training run from this map alone.** The next focused deliverable should be an operational Level 2 protocol, with proposed manuscript qualifications, before implementation. It should resolve these specific gaps:

1. Define each proposition separately: optimization stability of which model/trajectory; temporal utility measured by which per-level errors and comparisons; and generalization beyond which fitted data and prior selection decisions. Avoid using one pooled loss to stand for all three.
2. Declare primary versus diagnostic endpoints, sampling/temporal units, uncertainty interpretation, fixed compute and stopping rules, and any decision criteria before collecting new outcomes. Persistence and fitting-only mean baselines are justified proposed comparators, but must be identified as additions to the current protocol.
3. Document a confirmation-data strategy in light of repeated validation inspection. Verify raw-window overlap and encoder training exposure, not only row separation. Do not silently consume later evaluation partitions reserved for other scientific questions.
4. Resolve checkpoint identity for any inherited Level 1 evidence. A new checkpoint does not inherit a historical checkpoint's scientific results merely because the architecture and configuration names match.

This keeps the next decision tied to the manuscript's actual learning-process claim. It does not prescribe a new architecture, learning rate, gate, or longer training budget, and does not discard the completed negative results.

## Source fingerprints

SHA256 values identify the exact source bytes used for this map:

- Manuscript `cas-sc-template.tex`: `ad800c06db49c42d561a92a61642576986e327fcde998770f0c124e100b80d41`.
- Specification `implementation specification(3).tex`: `b1c6afd2f42076f725d87666ace382b4f05189d832914471d448c92fa9556804`.
- Consolidated `HDEG_Level2_Evidence_Boundary.md`: `f8c8c4b7c84efc42d471b8967507671895387f5f53f81cef644263e08f6413f3`.

Source line references are to these LaTeX files, not PDF page numbers. The companion boundary contains the eight run IDs, protocol IDs, and report hashes. Source and checkpoint hashes provide provenance, not scientific acceptance.
