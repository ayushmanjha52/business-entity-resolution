# Doppelgänger: business entity resolution that catches look-alikes

Given business records from three noisy sources, Doppelgänger finds every Source-2/3 record that belongs to each Source-1 entity (ML Challenge 2026, scored by macro F0.5).

Similarity alone fails on this data. The generator plants **sibling decoys**: records with the same name and street but a house number moved a few doors. It also disguises **genuine** records as pseudo-words, URLs, native-script names, aliases or empty addresses. Doppelgänger judges each record by the noise pattern that produced it and against its **rivals**, then builds each entity's list to maximise expected F0.5.

## Results
Source-1 training entities are split by a hash of their id: TRAIN 50 % (fitting), VALID 10 % (every choice), TEST 10 % (scored once per frozen design). The test split is decoy-heavier than train, measured without labels: hard same-street siblings per entity US 1.04 → 1.76, India 0.64 → 1.04. So VALID and TEST are scored **at test-like density**, adding calibrated synthetic sibling decoys (`train_aug`). Details: [docs/VALIDATION.md](docs/VALIDATION.md).

| System | VALID, test-like | **TEST, test-like** |
|---|---|---|
| Tuned fuzzy matcher | – | 0.6529 |
| v3 model (trained at train density) | 0.9552 | – |
| **v4: trained at test-like density** (2-stage LightGBM + ownership + expected-F0.5 lists) | **0.9754** | **0.9752** |

- TEST (v4): precision 0.9921, recall 0.9462, F1 0.9686, ROC-AUC 0.9986. US 0.9794, India 0.9688.
- The same v4 model at plain train density scores 0.9783 on VALID; v3 scored 0.9806 there but 0.9552 at test-like density.
- Candidate generation: pair recall 0.9794 on all 2.2M training entities; a perfect matcher on these candidates would score 0.9929.
- France has no labels. Its model trains on synthetic French records plus a self-training round on confidently decided real French test pairs. That approach, validated with India treated as unlabelled, gave 0.9392 → 0.9424.
- **99 % is not reached** on any held-out measurement. What remains:
  - true matches blocking never proposes: 0.007;
  - hard same-street siblings;
  - empty-address records whose name is shared by several Source-1 entities.

Test submission: 1,732,544 rows, 5,729,248 matched ids, official validator **PASS** with `--check-ids`.

## Quick start
```bash
python -m venv .venv
.venv\Scripts\activate            # Windows  (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt
set PYTHONPATH=src                 # Linux/macOS: export PYTHONPATH=src
```
Put the challenge data at `student_resource/dataset/{train,test}/` (or edit `paths.data_dir` in `configs/default.toml`). Paths are relative to this folder, so in the submission zip that means `code/business_entity_resolution/student_resource/dataset/`. Copy `student_resource/utils/validate_submission.py` there too if you want the end-to-end test to also run the official validator (it is skipped otherwise).

```bash
python -m doppelganger all         # prepare → block → train → validate → test → final → predict
```
or stage by stage. Every stage caches its output in `artifacts/` and later stages reuse it:

| Command | Output |
|---|---|
| `python -m doppelganger prepare` | normalised parquet + lexicon learned from TRAIN-role labels (`artifacts/base`, `artifacts/final`, `artifacts/lexicon.json`) |
| `python -m doppelganger block` | candidate pairs + recall/cost curve and oracle ceiling (`artifacts/pairs_*.parquet`); the calibrated `train_aug` decoys are built on first use by `train` |
| `python -m doppelganger train` | development stage-1/stage-2 LightGBM on TRAIN entities (`artifacts/models/`) |
| `python -m doppelganger validate` | decision floor chosen on VALID and frozen (`artifacts/decision.json`); VALID report |
| `python -m doppelganger test` | one-off TEST report with the frozen design (`artifacts/validation/*_test.*`) |
| `python -m doppelganger final` | models refit on every labelled entity, + synthetic France block (`artifacts/models_final*`) |
| `python -m doppelganger predict` | `output/matching_results.tsv`, `output/candidate_pairs.tsv` (the exact pairs the final model scored) |
| `python scripts/compare_models.py` | logistic regression / random forest / ExtraTrees / LightGBM on VALID pairs |
| `python -m doppelganger loco` | leave-India-out test of the synthetic-data approach |
| `python -m doppelganger evidence` / `explore` | evidence store and explorer at http://127.0.0.1:8765 |

Check the submission and run the tests:
```bash
python student_resource/utils/validate_submission.py --matching output/matching_results.tsv \
       --candidate output/candidate_pairs.tsv --test-dir student_resource/dataset/test --check-ids
python -m pytest -q
python scripts/create_submission.py --team <team_name>   # builds <team>_submission.zip
```

## How it works
1. **Normalise** (vectorised pyarrow):
   - split alias markers;
   - flag URL, handle, junk, native-script and pseudo names;
   - fold accents while keeping Indic vowel signs;
   - read digits inside words as letters (`c0mmerce` → `commerce`);
   - segment URL names into Source-1 words (`impexprivate.com` → `impex`);
   - extract house numbers into an int32 matrix.
2. **Learn a lexicon from TRAIN labels only.** State and département equivalences (`TN`/`தமிழ்நாடு` → Tamil Nadu, `Nord` → Hauts-de-France, the latter from unlabelled test pseudo-pairs) and a 1,312-token transliteration dictionary.
3. **Block** with 8 integer-packed key families in a sorted index. Each record keeps its best entities by rarity-weighted votes (top 8, ≥ 35 % of its best). The families are:
   - name×number, street×number, name pair, name×street, space-free name;
   - single name token;
   - street pair;
   - digit-deleted Source-1 numbers.
4. **Stage 1** LightGBM on 37 pairwise features:
   - name and address similarity;
   - rarity-weighted token overlap;
   - how many Source-1 entities share the name;
   - house-number geometry, including digit deletions;
   - noise flags.
5. **Stage 2** adds **competition features** from out-of-fold stage-1 scores (rank and margin among the entity's candidates, the rival entity's claim on the record) and **group-coherence features** (the record compared with the entity's most confident other record).
6. **Decide.** Each record goes to at most one entity. Each entity's list is the probability-ranked prefix that maximises expected F0.5. The probabilities are calibrated, and the only free parameter (a probability floor) is frozen on VALID.
7. **France without labels.** The reverse-engineered noise generator is re-applied to unlabelled French Source-1 records to synthesise labelled true variants and sibling decoys. France pairs are scored by the model trained with this block.

More detail: [docs/METHODOLOGY.md](docs/METHODOLOGY.md) · [docs/BLOCKING.md](docs/BLOCKING.md) · [docs/FEATURES.md](docs/FEATURES.md) · [docs/MODEL.md](docs/MODEL.md) · [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) · [docs/EXPERIMENT_LOG.md](docs/EXPERIMENT_LOG.md)

## Benchmark (v3 run, stage logs in `artifacts/run_*.log`)
Laptop with 16 GB RAM, 16 logical CPUs at 2.3 GHz, no GPU.

| Command | Work | Wall time | Peak RSS |
|---|---|---|---|
| `prepare` | lexicon + finalise 22.3M records (base tables reused) | 743 s | 7.3 GB |
| `block` | train 18.6M pairs + recall report, test 20.7M pairs | 4,301 s | 12.0 GB |
| `train` | features (37) for 18.6M pairs; stage 1 (2-fold OOF) + stage 2 on 9.3M TRAIN pairs | 1,439 s | 10.0 GB |
| `validate` | decision grid + 5 systems × 12 slices on VALID | 483 s | 6.9 GB |
| `test` | same on TEST | 329 s | 6.9 GB |
| `final` | refit on 18.6M labelled pairs; France synthetic block (830k records) + its model | 3,895 s | 12.9 GB |
| `predict` | features + routed scoring of 20.7M test pairs, lists, 2 TSVs | 770 s | 11.2 GB |

End to end, about 3.3 hours of CPU. First normalisation pass (`artifacts/base`), when absent, adds about 8 minutes.

## Repository
```
configs/default.toml        every tunable (paths, blocking, split, model, decision, synth)
src/doppelganger/           pipeline package (see docs/ARCHITECTURE.md)
src/doppelganger/explorer/  FastAPI app + single-page UI
scripts/                    create_submission.py, compare_models.py
tests/                      unit tests on small fixtures, explorer API test, end-to-end test
docs/                       methodology, blocking, features, model, validation, experiment log
```

## Licences and rules
Only permissive open-source libraries are used (LightGBM MIT, RapidFuzz MIT, pandas/numpy/pyarrow/scikit-learn BSD/Apache, FastAPI MIT). There is no pretrained language model. No external data, APIs or geocoding are used: every equivalence is learned from the provided files.
