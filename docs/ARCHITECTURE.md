# Architecture

```
student_resource/dataset (read-only)
   │  io.py             pyarrow TSV reader (tabs, no quoting, all strings)
   ▼
normalize.base_normalize  ─ chunked (1M rows), vectorised pyarrow.compute ─►  artifacts/base/*.parquet
   │   names: alias extraction, URL/handle/junk/native-script flags, legal-token split, dotted suffixes
   │   addresses: comma components, PO-box/null removal, first 6 digit-runs as an int32 matrix
   ▼
lexicon.fit  (train labels + unlabelled test structure)                    ─►  artifacts/lexicon.json
   │   component equivalences per country (TN→tamil nadu, தமிழ்நாடு→tamil nadu, gironde→nouvelle aquitaine)
   │   native-script token dictionary, localities, Source-1 vocabulary (pseudo-name detection)
   ▼
normalize.finalize ─────────────────────────────────────────────────────►  artifacts/final/*.parquet
   ▼
blocking.generate  (per country; integer keys; sorted index + searchsorted) ─►  artifacts/pairs_{split}.parquet
   │   NAME×NUM, ADDR×NUM, NAME×NAME, NAME×ADDR, NOSPACE keys; keys shared by >30 S1 records dropped;
   │   per S2/S3 record: top-8 S1 entities by rarity-weighted votes, ≥35 % of the record's best vote
   ▼
features.pair_features  (rapidfuzz cpdist, numpy number geometry, flags)   ─►  artifacts/feats_{split}.parquet
   ▼
model  stage 1: LightGBM on pairwise evidence (2-fold out-of-fold by entity)
   ▼
features.competition_features  (rank / margin / look-alike count among rivals, from stage-1 scores)
   ▼
model  stage 2: LightGBM on pairwise + competition features            ─►  artifacts/models/stage{1,2}.txt
       (a second model set, artifacts/models_unlabelled/, adds the synthetic block; pipeline.score routes each
        country's contiguous block of pairs to the model set matching whether that country had labels)
   ▼
decide.owner   (each S2/S3 record → at most one S1 entity)
decide.select  (per-entity prefix maximising expected F0.5; empty list when P(no match) wins)
   ▼
output/matching_results.tsv, output/candidate_pairs.tsv (validated with utils/validate_submission.py)
   ▼
evidence.build ─► artifacts/evidence*.parquet ─► explorer (FastAPI + one static page, SHAP on demand)
```

Self-supervision for unlabelled countries (France) runs beside the main path:
`selfsup.build_profile` (operator rates from labelled countries + vocabulary from the target's unlabelled records)
→ `synth.Generator` (noisy true variants + sibling decoys) → the same normalise → block → featurise path
→ appended to the training data of both stages of the *unlabelled-country* model set.

## Modules (`src/doppelganger/`)
| Module | Responsibility |
|---|---|
| `config.py` | loads `configs/default.toml` (all paths, caps, model and decision settings) |
| `io.py` | raw TSV reading, parquet caches, ground-truth alignment, submission writers |
| `normalize.py` | vectorised name/address normalisation and noise-operator flags |
| `lexicon.py` | data-learned equivalences (no external gazetteer) |
| `blocking.py` | integer-key inverted index and candidate generation |
| `features.py` | pairwise features and competition features |
| `model.py` | LightGBM training, out-of-fold scoring, importance |
| `decide.py` | ownership constraint and expected-F0.5 list selection |
| `metrics.py` | challenge metric (macro F0.5 incl. singletons) and slice reports |
| `synth.py` | noise generator (rates, vocabulary, generation) |
| `selfsup.py` | synthetic training block and the leave-one-country-out experiment |
| `pipeline.py` | stage orchestration and caching |
| `evidence.py` | explorer evidence store |
| `explorer/` | FastAPI app and the single-page UI |

## Memory and caching
- Every stage writes a parquet artefact to `artifacts/`; later stages read only the columns they need.
- Normalisation and features are chunked; blocking runs one country at a time and one million Source-2/3 records per chunk.
- Keys are packed `int64` (family « 58 | token id « 29 | number), so the index is a sorted numpy array rather than Python dictionaries.
- Peak resident memory is recorded by a sampling thread and written to `artifacts/benchmark.json`.

## Leakage boundaries
- Development models train on TRAIN entities (50 % of Source 1); VALID (10 %) drives every choice; TEST (10 %) is scored once. The submission models are refit on all labelled entities.
- Baseline thresholds are tuned on training entities.
- Stage-2 competition features for training rows are built from out-of-fold stage-1 scores.
- Test data is used only as unlabelled input: the France lexicon comes from identical-name pseudo-pairs, and synthetic seeds and vocabulary come from test records. No test labels exist.
