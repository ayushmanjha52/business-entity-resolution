# Validation

## Protocol
Source-1 training entities get a deterministic role from a hash of their id (`pipeline.split_roles`):

| Role | Share | Entities | Used for |
|---|---|---|---|
| TRAIN | 50 % | 1,103,066 | fitting the development models |
| VALID | 10 % | 220,929 | every choice: features, blocking, model, decision floor |
| TEST | 10 % | 220,804 | scored once per frozen design, never used for a choice |
| unused | 30 % | 662,022 | kept out of development; used by the final refit |

- Splitting is by Source-1 entity, so all records of one business are in one role, and each Source-2/3 record belongs to exactly one entity.
- The lexicon is learned from TRAIN-role labelled pairs only.
- Candidates and competition features are computed over all entities, so evaluated entities face realistic rivals.
- The metric is the challenge metric: F0.5 per Source-1 entity, averaged, singletons included.

### Test-like density (v4)
The test split is decoy-heavier than train. This was measured without labels on the candidate sets:

| per Source-1 entity | train | test |
|---|---|---|
| Source-2/3 records | 4.68 | 5.75 |
| hard same-street siblings (name ≥ 85, address ≥ 90, house number not kept), US | 1.04 | **1.76** |
| hard same-street siblings, India | 0.64 | **1.04** |
| same name at a different address, US / India | 0.78 / 0.94 | 0.78 / 1.07 |

True matches per entity are the same (~3.4). The surplus is hard same-street siblings.

`train_aug` = train + synthetic sibling decoys from the measured noise model, thinned per country to close exactly that hard-sibling gap (kept: US 93.8 %, India 52.3 % of 2.38M generated; 1.84M used; `artifacts/aug_calibration.json`). VALID and TEST are evaluated on `train_aug`, which is **test-like**. Synthetic decoys are never labelled true.

Reproduce: `python -m doppelganger validate`, then `python -m doppelganger test` (reports in `artifacts/validation/`).

## 1. Candidate generation (all 2.2M training entities, real records)
| | v2 | **v3/v4** |
|---|---|---|
| Train candidate pairs | 18,503,975 | **18,568,770** |
| Pair recall | 0.9605 | **0.9794** |
| Oracle macro F0.5 (perfect matcher on these candidates) | 0.9846 | **0.9929** |
| Test candidate pairs | 19,756,139 | **20,480,232** |

Details: [BLOCKING.md](BLOCKING.md).

## 2. Why the old validation over-stated the test score
The two-stage model trained on plain train, scored on VALID at two densities:

| VALID scored at | macro F0.5 | precision | recall | singleton false merges |
|---|---|---|---|---|
| train density (8.4 candidates / entity) | 0.9806 | 0.9948 | 0.9556 | 2.3 % |
| **test-like density** (9.7 / entity) | **0.9552** | 0.9645 | 0.9565 | 11.3 % |

The model accepted hard siblings once candidate groups got crowded. This is consistent with the leaderboard (~0.970) sitting well below the plain-VALID number.

## 3. The fix: train at test-like density
| Model | VALID, train density | **VALID, test-like** | **TEST, test-like** |
|---|---|---|---|
| trained on plain train (v3) | 0.9806 | 0.9552 | – |
| **trained on train_aug (v4)** | 0.9783 | **0.9754** | **0.9752** |

A gain of +0.020 where the test data actually sits, for −0.002 at the density the test does not have.

### TEST in detail (test-like, 220,804 entities, frozen design)
| System | macro F0.5 |
|---|---|
| B0 fuzzy rule, tuned on TRAIN (name ≥ 85, address ≥ 70) | 0.6529 |
| B1 stage-1 model, global threshold 0.7 | 0.9707 |
| B2 stage-1 + ownership + expected-F0.5 lists | 0.9720 |
| B3a stage-2, threshold 0.5 | 0.9729 |
| **FINAL stage-2 + ownership + expected-F0.5 lists** | **0.9752** (US 0.9794, India 0.9688) |

| Pair level (candidate pairs of TEST entities, synthetic decoys included) | value |
|---|---|
| Candidate pairs | 2,144,787 |
| Blocking recall (real true pairs retained) | 0.9792 |
| Predicted pairs | 727,929 |
| TP / FP / TN | 722,171 / 5,758 / 1,391,647 |
| FN (in candidates + lost in blocking) | 41,054 (25,211 + 15,843) |
| Precision / recall / F1 | 0.9921 / 0.9462 / 0.9686 |
| Accuracy over candidate pairs | 0.9856 |
| ROC-AUC / PR-AUC | 0.9986 / 0.9969 |
| Singleton false-merge rate | 3.8 % |

Recall counts true pairs lost in blocking as misses.

### Decision threshold
Each record goes to its best entity, and each entity keeps the probability-ranked prefix that maximises expected F0.5. The only free parameter, a probability floor, was searched on test-like VALID over 0–0.5 and frozen at 0.5 (`artifacts/decision.json`). It is inert (0.9754 at every value) because the probabilities are calibrated.

## 4. Model comparison (stage-1 features, 1M TRAIN pairs → VALID pairs)
`python scripts/compare_models.py --train-pairs 1000000`

| Model | ROC-AUC | PR-AUC | log-loss | best F1 |
|---|---|---|---|---|
| Logistic regression (standardised) | 0.9768 | 0.9702 | 0.1911 | 0.9015 |
| Random forest, 200 trees | 0.9973 | 0.9964 | 0.0674 | 0.9711 |
| ExtraTrees, 200 trees | 0.9964 | 0.9950 | 0.0859 | 0.9654 |
| LightGBM 63 leaves, 400 trees | 0.9987 | 0.9982 | 0.0444 | 0.9791 |
| **LightGBM 127 leaves, lr 0.06, 600 trees (used)** | 0.9987 | 0.9983 | 0.0428 | 0.9798 |
| LightGBM 255 leaves, 800 trees | 0.9988 | 0.9983 | 0.0428 | 0.9799 |
| XGBoost (lossguide, 127 leaves, lr 0.06, 600 trees) | 0.9987 | 0.9982 | 0.0429 | 0.9797 |
| CatBoost (depth 8, lr 0.08, 1000 trees) | 0.9986 | 0.9981 | 0.0448 | 0.9790 |

XGBoost ties LightGBM and CatBoost trails it; neither justifies a rebuild. Both are optional: `scripts/compare_models.py` includes them only when installed (`pip install xgboost catboost`, Apache-2.0).

## 5. Experiments that were measured and rejected
| Change | Measured | Decision |
|---|---|---|
| Third stage (rivals re-read from out-of-fold stage-2 scores) | +0.0009 in one run, −0.0005 in the next | not robust, not used (`model.stages = 2`) |
| Stop words `n`, `d`, `l` (French cleanup) | −0.0008 on VALID: `n` is the canonical form of "North"; `d`/`l` are initials and block letters | removed; genuinely French words kept |
| Decision sums in float32 | bug: per-entity prefix sums lost precision inside a running sum over millions of pairs | fixed (float64), +0.0001 |
| Decoys at the raw record-count gap (+1.07 / entity) | too many hard siblings: VALID 0.9466 | replaced by the calibrated density |

## 6. Countries without labels (France)
Self-training was validated by treating India as unlabelled (US labels + synthetic India, scored on India VALID):

| | India VALID macro F0.5 | precision | recall | singleton false merges |
|---|---|---|---|---|
| B: US labels + synthetic India | 0.9392 | 0.9746 | 0.8950 | 8.8 % |
| **S: + self-training on real India pairs (no India labels)** | **0.9424** | 0.9774 | 0.8985 | 7.9 % |
| real India labels (upper bound) | 0.9727 | | | |

France uses S: synthetic French records (decoy density taken from the test split), then a second round on real French test pairs pseudo-labelled by the first-round model (entities whose every candidate is decided with p ≥ 0.95 or ≤ 0.05).

## 7. Error analysis (VALID)
- **Blocking (0.007 of F0.5).** Empty-address records whose name is shared by many Source-1 entities, and native-script names outside the learned transliteration.
- **Hard siblings (addressed by v4).** Same name and street, shifted house number, in crowded candidate groups.
- **Name-only records.** For empty-address records whose exact core name belongs to one Source-1 entity, that entity is the true match 97.2 % of the time. When 2–20 entities share the name, the data rarely identifies the right one.

## 8. Leaderboard, and the final decision layer
| File | Local held-out | Leaderboard |
|---|---|---|
| v3 (trained at train density) | VALID 0.9806 plain / 0.9552 test-like | ~0.970 |
| v4 (trained at test-like density) | VALID 0.9783 plain / 0.9754 test-like | 0.9709 |
| **v5 = v4 + cross-source fill** | VALID 0.97862 plain / 0.97544 test-like | expected ~0.971 |

The leaderboard moved much less than the test-like measurement predicted. So the real test is closer to plain density: the synthetic siblings are harder than the real ones.

Decision-layer experiments on the development models (VALID chooses, TEST confirms):

| Change | Plain VALID | Plain TEST | Decision |
|---|---|---|---|
| Cross-source fill, p ≥ 0.5 | 0.97826 → 0.97862 | 0.97816 → 0.97856 | **kept** |
| Second-chance ownership | −0.00007 | −0.00004 | rejected |
| One record per source per entity | – | – | rejected: 51-55 % of entities truly have several |
| Per-segment probability shifts (country × address × name frequency) | +0.00043 | +0.00020, shifts reverse between densities | rejected: not robust |

Blocking follow-up: 86.6 % of the remaining misses share a name token with their entity, and nearly all with an address share address tokens. They are ranked out of the top 8, not unretrievable. Re-ranking by address evidence lowered recall. A top-12 cut raised recall 0.9794 → 0.9814 and test-like VALID/TEST by +0.0003/+0.0004, but its final refit exceeds 16 GB of RAM (see EXPERIMENT_LOG #49).

## 9. Not measurable
- France accuracy (no labels).
- The leaderboard score (only the portal computes it).
