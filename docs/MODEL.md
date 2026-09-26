# Models and decision layer

Code: `src/doppelganger/model.py`, `pipeline.train`, `decide.py`. Settings: `[model]`, `[decision]`, `[synth]`.

## Why LightGBM
- The inputs are 50 tabular features (37 pairwise + 9 competition + 4 group-coherence) over 18.6M pairs, on CPU only.
- Gradient-boosted trees handle NaN-heavy number features natively and train in minutes.
- They produce calibrated probabilities (verified below) and exact SHAP contributions for the explorer.
- MIT licence and far below the 8B-parameter limit: no neural or pretrained model is used.

Hyper-parameters: 127 leaves, learning rate 0.06, 600 trees, min 100 samples per leaf, 0.7 row / 0.8 column subsampling, seed 42. Chosen on VALID from the comparison in [VALIDATION.md](VALIDATION.md) §3 (logistic regression, random forest, ExtraTrees, three LightGBM sizes).

## Training data and leakage control
- Source-1 entities are split deterministically by a hash of the id: **10 % VALID** (all selection), **10 % TEST** (scored once, at the end), **50 % TRAIN** for the development models, 30 % unused in development. Blocking and competition still see all entities.
- The **final** models (`python -m doppelganger final`, `artifacts/models_final*`) are refit on every labelled entity with the frozen design; they produce the submission.
- Stage 1 is trained on the training entities. Its scores for those rows are **2-fold out-of-fold, split by entity**, so the stage-2 competition features for training rows never come from a model that saw the same entity.
- Stage 2 is trained on pairwise + competition features of the same rows. Holdout and test rows get stage-1 scores from the full stage-1 model.
- Baseline thresholds are tuned only on training entities.

## Two model sets (per-country routing)
| Model set | Trained on | Scores |
|---|---|---|
| `artifacts/models_final/` | real labels (all US + India entities) | pairs whose country has training labels |
| `artifacts/models_final_unlabelled/` | the same + synthetic block for each country without labels (France: 830k generated records, 1.75M pairs) | pairs of unlabelled countries |

Candidate pairs never cross countries, so routing does not mix competition groups. The reason for routing is measured: one shared model with the synthetic block lost 0.0009 on the labelled holdout (see EXPERIMENT_LOG #9).

## Calibration (holdout, first full model)
| predicted bin | 0.3–0.4 | 0.4–0.5 | 0.5–0.6 | 0.6–0.7 | 0.8–0.9 | 0.95–1 |
|---|---|---|---|---|---|---|
| mean predicted | 0.348 | 0.447 | 0.547 | 0.650 | 0.857 | 0.997 |
| observed rate | 0.357 | 0.476 | 0.547 | 0.638 | 0.859 | 0.998 |

## Decision layer
1. **Ownership** (`decide.owner`): each Source-2/3 record keeps only the pair where its probability is highest. In the ground truth, 0 of 7,638,365 matched records belong to two entities.
2. **Expected-F0.5 list** (`decide.select`): candidates are sorted by probability. For each prefix size k, `E[F0.5] ≈ 1.25·Σ_{i≤k} p_i / (k + 0.25·Σ_i p_i)`. The empty list scores `Π(1 − p_i)`, the probability that the entity is a singleton among its candidates. The best option wins.
   - A Monte-Carlo estimate of the exact expectation gave 0.9676 vs 0.9677 for the closed form.
   - Probability floor: searched on VALID over 0–0.5 and frozen at 0.5 (`artifacts/decision.json`). It is nearly inert (0.98052 → 0.98056) because the probabilities are calibrated.

3. **Cross-source fill** (`decide.cross_source_fill`, `decision.cross_source_min_prob = 0.5`): 80 % of labelled entities are matched in both Source 2 and Source 3 and only 14 % in exactly one, so a list that holds a single source usually misses a record. The best still-unassigned record of the missing source joins the list when p ≥ 0.5 (chosen on VALID; VALID 0.97826 → 0.97862, TEST 0.97816 → 0.97856 at plain density).
   - Rejected alternatives: second-chance ownership (−0.00007) and one record per source (false for half the entities).

## Artefacts
`artifacts/models*/stage{1,2}.txt` (LightGBM text format), `artifacts/scores_train*.parquet` (p1, p2, label, role per train pair), `artifacts/scores_test.parquet`.
