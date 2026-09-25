# Validation

## Protocol
Source-1 training entities get a deterministic role from a hash of their id (`pipeline.split_roles`):

| Role | Share | Entities | Candidate pairs | Used for |
|---|---|---|---|---|
| TRAIN | 50 % | 1,103,066 | 9,289,987 | fitting the development models |
| VALID | 10 % | 220,929 | 1,851,194 | every choice: features, blocking, model, hyper-parameters, decision floor |
| TEST | 10 % | 220,804 | 1,864,156 | scored **once**, after the design was frozen |
| unused | 30 % | 662,022 | 5,572,622 | kept out of the development run; used by the final refit |

- Splitting is by Source-1 entity, so all records of one business sit in one role. Each Source-2/3 record belongs to exactly one entity.
- The lexicon (transliteration, address equivalences) is learned from TRAIN-role labelled pairs only. The earlier version sampled all labelled pairs, which let validation labels leak into normalisation.
- Candidates and competition features are computed over all entities, so evaluated entities face realistic rivals.
- The metric is the challenge metric: F0.5 per Source-1 entity, averaged, singletons included.
- The final submission model is refit on every labelled entity (`python -m doppelganger final`), using the design frozen above.

Reproduce: `python -m doppelganger validate` (tables in `artifacts/validation/*_valid.*`), then `python -m doppelganger test` (`*_test.*`).

## 1. Candidate generation (all 2.2M training entities)
| | v2 | **v3** |
|---|---|---|
| Train candidate pairs | 18,503,975 | **18,577,959** |
| Pair recall (true matches retained) | 0.9605 | **0.9794** |
| Oracle macro F0.5 (perfect matcher on these candidates) | 0.9846 | **0.9929** (US 0.9963, India 0.9877) |
| Test candidate pairs | 19,756,139 | **20,700,123** (11.9 per Source-1 entity, 2.08 per Source-2/3 record) |
| Reduction ratio vs all same-split cross pairs | 0.99999886 | 0.99999880 |

Where the recall came from: [BLOCKING.md](BLOCKING.md).

## 2. Held-out results (frozen design)
| System | VALID | **TEST** |
|---|---|---|
| B0 fuzzy rule, tuned on TRAIN (name ≥ 85, address ≥ 70) | 0.7351 | 0.7347 |
| B1 stage-1 model, global threshold 0.7 (tuned on TRAIN) | 0.9758 | 0.9757 |
| B2 stage-1 + ownership + expected-F0.5 lists | 0.9765 | 0.9767 |
| B3a stage-2, threshold 0.5 | 0.9790 | 0.9790 |
| **FINAL stage-2 + ownership + expected-F0.5 lists** | **0.9806** | **0.9806** |
| v2 (previous submission), same VALID entities | 0.9678 | – |

VALID and TEST agree to the fourth decimal, so the gain is not an artefact of selecting on VALID.

### Pair-level report, TEST (220,804 entities)
| | value |
|---|---|
| Candidate pairs of TEST entities | 1,864,156 |
| True pairs / retained by blocking | 763,225 / 747,405 (blocking recall 0.9793) |
| Predicted pairs | 731,490 |
| TP / FP / TN | 728,070 / 3,420 / 1,113,331 |
| FN (in candidates + lost in blocking) | 35,155 (19,335 + 15,820) |
| Precision / recall / F1 | 0.9953 / 0.9539 / 0.9742 |
| Micro F0.5 | 0.9868 |
| Accuracy over candidate pairs | 0.9878 |
| ROC-AUC / PR-AUC (stage-2 probability) | 0.9992 / 0.9985 |
| **Macro F0.5 (challenge metric)** | **0.9806** (US 0.9852, India 0.9736, singletons 0.9754) |

Recall counts true pairs lost in blocking as misses. "Accuracy" is reported for completeness only; with about 1.5 negatives per positive in the candidate set it says little.

### Decision threshold
The decision is not one global threshold. Each record goes to its best-scoring entity, and each entity keeps the probability-ranked prefix that maximises expected F0.5. The only free parameter is a probability floor under that rule, searched on VALID and frozen in `artifacts/decision.json`:

| floor | 0.0–0.4 | **0.5** |
|---|---|---|
| VALID macro F0.5 | 0.98052 | **0.98056** |

The floor is almost inert because the probabilities are calibrated. For comparison, a plain global threshold with ownership (TEST):

| threshold | 0.3 | 0.5 | 0.6 | 0.7 | 0.8 | 0.9 |
|---|---|---|---|---|---|---|
| macro F0.5 | 0.9748 | 0.9791 | 0.9801 | 0.9803 | 0.9799 | 0.9777 |
| pair precision | 0.9801 | 0.9906 | 0.9931 | 0.9947 | 0.9962 | 0.9976 |
| pair recall | 0.9689 | 0.9620 | 0.9592 | 0.9563 | 0.9519 | 0.9431 |

The expected-F0.5 lists (0.9806) beat the best global threshold (0.9803).

## 3. Model comparison (stage-1 features, 1M TRAIN pairs → VALID pairs)
`python scripts/compare_models.py --train-pairs 1000000`

| Model | ROC-AUC | PR-AUC | log-loss | best F1 | fit |
|---|---|---|---|---|---|
| Logistic regression (standardised) | 0.9768 | 0.9702 | 0.1911 | 0.9015 | 4 s |
| Random forest, 200 trees | 0.9973 | 0.9964 | 0.0674 | 0.9711 | 51 s |
| ExtraTrees, 200 trees | 0.9964 | 0.9950 | 0.0859 | 0.9654 | 43 s |
| LightGBM 63 leaves, lr 0.08, 400 trees (v2 setting) | 0.9987 | 0.9982 | 0.0444 | 0.9791 | 23 s |
| **LightGBM 127 leaves, lr 0.06, 600 trees (used)** | 0.9987 | 0.9983 | 0.0428 | 0.9798 | 43 s |
| LightGBM 255 leaves, lr 0.05, 800 trees | 0.9988 | 0.9983 | 0.0428 | 0.9799 | 78 s |

The 255-leaf model gains 0.0001 F1 for twice the cost, so the 127-leaf setting is kept. XGBoost and CatBoost were not installed and were not tried; LightGBM represents gradient boosting. No stacking: the tree ensembles are all dominated by LightGBM, so averaging them would add cost without an expected gain.

## 4. Error analysis (VALID)
| Counterfactual | macro F0.5 |
|---|---|
| actual | 0.9806 |
| remove all 3,300 false merges | 0.9849 |
| recover all 19,456 in-candidate misses | 0.9886 |
| perfect matcher on the candidates (oracle) | 0.9929 |
| remaining blocking loss (1 − oracle) | 0.0071 |

- **Misses (in candidates).** 43 % are Source-2/3 records with an empty address, against 3.4 % of all true pairs. Their only evidence is the name, and many Source-1 businesses share generated names ("Bharat Foundation Private Limited" appears 59 times). 5,275 misses are records claimed by a rival entity with a higher score.
- **False merges.** Two families:
  - look-alike decoys, e.g. "Cancer Council, 341 vs 344 Greenfields Lane";
  - records whose true entity was never a candidate. The record then goes to the best look-alike, which is a blocking miss showing up as a false positive.
- **Blocking misses (15,764 on VALID).** Mostly empty-address records whose name is shared by more than 150 Source-1 entities, plus native-script names outside the learned transliteration.

## 5. Countries without labels
France cannot be measured, because no labels exist. The generator-inversion approach was validated earlier by treating India as unlabelled: US labels only scored 0.9098; adding synthetic India scored 0.9193; real labels scored 0.9513 (`python -m doppelganger loco`). That experiment predates v3 and was not re-run.

## 6. Not measurable
- France accuracy (no labels).
- Leaderboard score: only the portal computes it.
