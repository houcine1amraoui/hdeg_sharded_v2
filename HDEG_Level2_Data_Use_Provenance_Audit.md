# HDEG — Read-only Level 2 data-use and provenance audit

20 October 2026. Scope: submitted source, existing preflight/report records, and saved forecast selections. No training, inference, preprocessing, data reassignment, or source modification was performed.

## Decision

**No eligible fresh normal-operation confirmation panel is established by the available records.** This is an evidence-eligibility conclusion, not proof that no suitable data exist anywhere.

The existing checkpoint's training data cannot supply an unseen-to-the-encoder panel. Validation has been used repeatedly in development. Actor 2 and the later Actor 1 period have already been inspected in Level 1 and are not demonstrated to be untouched normal-operation confirmation data. They must not be silently repurposed.

Under the prospective Level 2 protocol, independent confirmation eligibility is therefore `NOT_ESTABLISHED`; the prospective G confirmation assessment remains `NOT_ASSESSED`. The completed validation diagnostic remains valid as a bounded, unfavorable held-out-from-training comparison. Repeated inspection limits future confirmatory claims; it does not erase that result or prove that validation rows were used for gradient updates.

## 1. Evidence strength and limitations

This audit distinguishes:

- **Direct local checks:** parsing saved reports, comparing their hashes/references, reading submitted code, and recomputing timestamp-support relations in the saved forecast-selection archive.
- **Reported execution evidence:** the preflight's full start-timestamp audit and checkpoint metadata. These are inspected records, not fresh raw-data checks performed here.
- **Code-path evidence:** what the submitted preprocessing/training code would do under the recorded configuration. Source alone cannot reconstruct an undocumented historical execution.
- **Unknown history:** original raw provenance, preprocessing-fit execution, feature-selection rationale, complete historical commands, and exposure of unselected observations.

Raw CSVs, complete processed timestamp arrays, the fitted scaler, and all window tensors were not available in the audited local materials. No GPU/model execution was performed. Source findings below must not be presented as a demonstrated quantitative effect on the saved tensors.

## 2. Data-use ledger

| Data | Documented extent/use | Eligibility for fresh confirmation |
|---|---|---|
| CU `train` | 1,632,930 pairs; checkpoint metadata records 1,632,930 processed samples, 200 shards, 51,030 batches. Subsequent frozen-encoder pilots and train-only probes reuse subsets. | **Ineligible for the existing encoder:** training exposure. An internal probe split does not remove it. |
| CU `val` / N1 | 181,410 pairs; Level 1 analyses; 2,048 selected pairs repeatedly used for Level 2 diagnostic/pilot evaluation. | **Development evidence.** Not established as fresh confirmation for the next adaptively chosen candidate. |
| `actor2_test` / A2 | 259,170 pairs in the R2 population. Included in R1/R2, R3 and qualitative PCA reports. Configured period November 8–10, 2022. | **Not certified eligible:** prior scientific inspection; normal-operation/event status unresolved; reserved evaluation role. No evidence here establishes gradient fitting on this split. |
| `actor1_test` / N2 | 597,557 pairs in the R2 population. Included in the same Level 1 analyses. Configured period November 11–17, 2022. | **Not certified eligible:** same concerns. Later time alone is insufficient. |
| Linear-probe evaluation group | Last 16 of the 32 selected training blocks, 1,024 pairs. | **Unseen to the fitted linear predictor, but seen by the encoder** and used in earlier diagnostics. Not independent complete-pipeline confirmation. |
| Unselected rows or another dataset | No complete exposure/normality/provenance ledger provided. | **Unknown**, not assumed eligible. This audit does not certify BRE or new data not represented in the records. |

The R2 report documents 100,000 sampled pairs with 181,716 unique endpoints drawn from the pooled N1/A2/N2 population of 1,038,137 windows. R3 documents 32 blocks × 128 windows per condition. PCA documents 128 windows per condition. These demonstrate scientific inspection of all three recording conditions; they do not demonstrate that every row was individually inspected or that any of these analyses optimized the HDEG parameters.

Prior Level 1 use is not synonymous with supervised label leakage. It nevertheless prevents certifying those periods as untouched confirmation data without a fuller account of selection decisions, disjointness, and study roles. A retrospective new subset is not automatically a new experiment independent of previous findings.

## 3. Temporal support findings

The V2 preflight reports these processed supports:

| Split | First input start | Last target end |
|---|---|---|
| Train | 2022-10-18 00:00:00 | 2022-11-05 21:35:59 |
| Validation | 2022-11-05 21:36:00 | 2022-11-07 23:59:59 |

Times are decoded from saved nanosecond values in the dataset timestamp coordinate; this audit does not establish the raw recording timezone. The one-second boundary separation is consistent with distinct raw timestamp support, not a temporal independence guarantee or an embargo of meaningful behavioral duration.

Preflight coverage is explicitly limited: all train/validation start timestamps and manifest entries were checked, but tensor values and target timestamp arrays were checked only in first/middle/last shards. Its full-boundary conclusion uses the regular-step target extent. This audit does not upgrade that into a new exhaustive check of all target arrays or corruption-free source tensors.

Direct recomputation from `selected_windows.npz` confirms:

- 2,048 selected pairs and 32 blocks in each of train and validation.
- Every selected pair's recorded target end is 30 seconds after its recorded input start, consistent with 30-observation windows shifted by one observation.
- Selected train support spans October 18 01:15:46 through November 5 10:39:27; selected validation support spans November 5 22:22:57 through November 7 22:26:55.
- Selected train and validation supports do not overlap. No adjacent selected blocks within either split overlap in their recorded input-through-target support.
- Within each block, successive pairs still overlap heavily; they are not independent samples.

The linear report documents no raw-support overlap between its fit and evaluation groups and a gap of 37,039 seconds (about 10.29 hours). This protects the linear fit/evaluation separation, not the encoder exposure boundary. The learning-rate calibration's existing record reports that all 256 probe windows share raw observations with update windows despite distinct row identities; it remains a local training diagnostic.

## 4. Preprocessing provenance

### Favorable code-path findings

`src/data_preparation/data_preparation.py::normalize` fits `MinMaxScaler` on `splits['train']` only, then transforms validation and both test splits. Chronological train/validation splitting occurs before window construction. The sharded window builder operates separately on each split; its construction does not intentionally create a window crossing split arrays.

The current recorded configuration uses CU only, filtering and cleaning enabled, a 0.1 validation ratio, one-second data, and window size 30. At `downsample_freq == '1s'`, the downsampling branch is skipped. The merged-home nearest-time join is also inactive. These inactive paths are not evidence of leakage in this run.

### Unresolved and adverse provenance findings

1. **Fitted scaler history is unverified.** The saved preflight explicitly states this. `save_processed_data` writes `scaler.pkl`, arrays, timestamps and device IDs, but the inspected routine does not write a raw-source hash, fitting-row ledger, or complete preprocessing execution manifest. Train-only fitting in the code is positive evidence of design intent, not proof of the saved scaler's execution history.
2. **A data-dependent column decision occurs before splitting.** With CU cleaning enabled, `clean_cu_data` calls `dropna(axis=1, how='all')` on the filtered full loaded dataframe. Therefore channel retention can depend on observations outside the training period. This is a preprocessing boundary concern under a strict unseen-pipeline definition. The actual retained set might equal a train-only decision; that equivalence has not been tested. It is not evidence that validation/test values were used to fit MinMax statistics, nor a quantified explanation of forecasting failure.
3. **Feature-list origin is unspecified.** Filtering reads `configs/CU_selected_features.txt` and intersects it with available columns. A predeclared device schema may be legitimate, but selection based on full-recording variability or evaluation outcomes would be different. The rationale and history are not established here.
4. **Continuity validation is incomplete in the window-builder source.** Its `validate_timestamps` checks dimensionality and length; that function does not enforce monotonicity, uniqueness, or a one-second step. Sliding by row therefore does not itself guarantee a continuous one-second window. The later train/validation preflight provides stronger recorded checks for those splits. Equivalent full-support guarantees for A2/N2 are not established by this audit.

Do not rerun preprocessing or overwrite artifacts to resolve these questions. First compare existing source/fitted-state provenance and timestamp summaries read-only.

## 5. Checkpoint and Level 1 links

The current V1 preflight, R1, R2, primary R3, qualitative report, and V2 preflight all identify checkpoint:

`4b036348b014652bb84e0f5b44a2b46374bae00213f4ad07b1d4746ff862157b`.

Direct SHA256 comparison verifies:

- Current V1 preflight hash `b30217b953c6ef82c57711488c375818c96aee58c4aec8e1c3a226bff4e39268` matches the V2 preflight reference and each inspected current Level 1 report's preflight reference.
- Current primary R3 report hash `192e703c638752a13bd52de177273a7c5482d70b1fad2190d14bb7f9210973f5` matches the R1, R2 and qualitative reference-report fields.

This resolves the report-level identity question for those current reports. It does **not** prove that every figure or numeric result in the October 18 manuscript was regenerated from them. The manuscript export predates these rerun report submissions; figure-to-report provenance remains a separate check, not a reason to discard the consistent current reports.

The earlier experiment manifest records the old Level 1 checkpoint `d47f7b...` as a legacy reference and explicitly states that it was not recovered. That historical manifest must not override the newer matching report chain. Its `PRESERVED` status describes a past action; this audit has not verified that its checkpoint copy still exists after output-protection cancellation.

The training source hash recorded by the V2 preflight, `550e38c7367699292eab1f6b46022fc11754fb7aea7d3d2b6dac8daa8a83f20e`, matches the supplied checkpoint-protection training-source version exactly. It does not match the older code-refresh training file. This identifies the source inspected at preflight time; it is not by itself proof of which source produced the checkpoint.

That matching source accepts a split argument and passes `train=True` in its training loop. Thus a split name alone is not a safeguard against optimizing on validation/test data. The recorded sample/shard/batch counts support full training-split use in this run, but the complete command/resume history is still needed to exclude additional undocumented fitting. No such extra fitting is alleged by this audit.

The preflight also warns that the checkpoint lacks original device IDs. Agreement of current device order and compatibility mask cannot independently exclude permutations within identical mask columns. Retain this provenance limitation.

## 6. Next decision and minimum missing evidence

The current evidence is sufficient to decline certifying a fresh confirmation panel. More inventory work should be targeted rather than another general preflight or training run.

For any proposed candidate period, a small read-only provenance bundle should contain:

- Global and per-split window manifests; complete timestamp support summaries, monotonicity/duplicate/gap counts, and preprocessing-boundary support. `timestamps.npz` would allow direct checks without sending the large window tensors.
- Existing scaler metadata (feature count, fit sample count, fitted minima/maxima, hash), device list, selected-feature list and its rationale; preprocessing config/source hashes, raw CSV identity and the actual preparation command/log if retained. Scaler attributes alone cannot prove which rows fitted it.
- Training command/resume history and any retained run configuration identifying split, caps, initialization and checkpoint digest.
- A short exposure ledger for the candidate period: prior fitting, diagnostic inspection, checkpoint/design selection, and intended Level 3/4 use; plus the basis for calling it normal-operation data.

These records may strengthen provenance or identify a genuinely separate eligible period. They cannot erase already documented exposure. Do not ask for another large code or checkpoint archive merely to answer data-use questions.

**Recommended research path with the evidence currently available:** retain train/validation as development data and the current Level 2 result as a bounded negative/exploratory outcome. Keep A2/N2 in their documented evaluation roles. If further optimization work is chosen later, label it exploratory and do not claim independent confirmation unless a qualifying panel is separately established. No new training experiment is chosen by this audit.

## Source register

Reviewed local copies of previously submitted artifacts:

- `upload/preflight_report.json`, `upload/v1 preflight.json`, `upload/experiment_manifest.json`.
- `upload/r1_report.json`, `upload/r2_report.json`, `upload/r3 primary_report.json`, `upload/qualitative_report.json`.
- Forecast diagnostic run `20261018T190648Z_a76a55fd`: report and selected-window archive.
- Linear predictability run `20261020T075653Z_9cafbce3`: report and partition metadata.
- Submitted code-refresh preprocessing, window preparation, configuration, and save routines; training-source version matched above.
- Companion Level 2 evidence boundary and operational protocol for earlier probe limitations and prospective eligibility rules.

The following source fingerprints were computed during this audit.

- `upload/preflight_report.json`: `ce2649e36b36f7fd74790228b3969f2290a24a8e76724d698f272cad43a58dba`.

- `upload/v1 preflight.json`: `b30217b953c6ef82c57711488c375818c96aee58c4aec8e1c3a226bff4e39268`.

- `upload/r1_report.json`: `7e7664b6744b256a297bf479c3e6709bce5dfc8ded0a4ba557114613750ed356`.

- `upload/r2_report.json`: `5ee9a75785834bf5fad7c6fd004e130a61898e9efb9e44eea940c9ba4e2550e9`.

- `upload/r3 primary_report.json`: `192e703c638752a13bd52de177273a7c5482d70b1fad2190d14bb7f9210973f5`.

- `upload/qualitative_report.json`: `00910331c474aea503443ef1337f69c807e09bfea963456bc6f4ca715144f5cd`.

- `audit_v2_forecast/20261018T190648Z_a76a55fd/selected_windows.npz`: `679db244c9cd532132f946524fca39726d13f81d9c9bb4bd284a973329e090f1`.

- `audit_refresh/code/src/data_preparation/data_preparation.py`: `a4d1d9495ef7f5c3af4324114e2d328e371d71589999ad8e67dd787da1e41739`.

- `audit_refresh/code/src/data_preparation/prepare_windows_sharded.py`: `d4084cb985fa6e0599b26c86ddef762f039c795db7c7da45a0960974f9a43657`.
