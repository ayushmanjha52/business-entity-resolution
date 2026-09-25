# Methodology — Doppelgänger (overview)

The dataset is produced by a noise generator, and two measured facts drive the whole design (`docs/RESEARCH_REPORT.md`):

1. **The traps are look-alikes.** Unmatched Source-2/3 records are mostly sibling copies of a real business: same street, near-identical name, house number shifted by a few units. Plain similarity scores them as matches.
2. **The safe matches look least alike.** Pseudo-word names (98 % true), aliases (100 %), URL names (96 %) and empty addresses (98 %) are almost always genuine. Plain similarity rejects them.

So each candidate is judged against **the noise operator that produced it** and against **its rivals**, not only against its reference record.

| Step | What it does | Details |
|---|---|---|
| 1. Normalise | Vectorised pyarrow. Splits alias markers; flags URL, junk, native-script and pseudo-names; folds Latin accents while keeping Indic vowel signs; collapses dotted legal forms; extracts up to 6 house numbers into an int32 matrix; canonicalises street abbreviations | `normalize.py` |
| 2. Learn a lexicon | From the provided files only: 211 component equivalences (India 48, e.g. `tn`/`தமிழ்நாடு` → tamil nadu; France 9, e.g. `nord` → hauts de france, learned from 400k **unlabelled** test pseudo-pairs), 1,312 transliteration tokens (93 % coverage), localities, Source-1 vocabulary | `lexicon.py` |
| 3. Block | 8 integer-packed key families, sorted index + `searchsorted`, rarity-weighted votes, top-8 / ≥35 % of best per record. Pair recall 0.9794 with 18.6M train pairs | [BLOCKING.md](BLOCKING.md) |
| 4. Pair features | 26 features: name and address similarity, **house-number geometry**, noise flags | [FEATURES.md](FEATURES.md) |
| 5. Stage 1 | LightGBM on pairwise evidence; 2-fold out-of-fold by entity | [MODEL.md](MODEL.md) |
| 6. Stage 2 (competition) | + 9 features describing each pair relative to its rivals (other records of the entity, other entities claiming the record) | [FEATURES.md](FEATURES.md) |
| 7. Decide | One owner per record, then per-entity list maximising expected F0.5 (probabilities are calibrated) | [MODEL.md](MODEL.md) |
| 8. Unlabelled countries | Generator inversion: synthetic labelled pairs from unlabelled French records; routed model for France | below |

## Generator inversion for France (`synth.py`, `selfsup.py`)
France has no labels. The leave-one-country-out test shows this matters: a model trained on US labels only scores 0.910 on India, against 0.951 with India labels.

The generator re-applies the reverse-engineered noise operators to unlabelled Source-1 records of the target country:
- **17 operator rates** are measured on labelled pairs of the other countries. Examples: pseudo-name 1.7 %, alias 2.1 %, URL 6.6 %, name typo 21.8 %, token drop 17.5 %, empty address 4.7 %, injected door number 19.6 %, address component drop 8.1 %, address typo 11.3 %.
- **Sibling decoys**: a descriptor word is added or swapped, and the house number is shifted by a delta sampled from the empirical shifts of real look-alike negatives. The number per entity is estimated from the target country's own record-to-entity ratio (France: 37 % of generated records are decoys).
- **Vocabulary only from the target's unlabelled records**: descriptors and legal forms from its Source-1 names; pseudo-names, alias markers, number prefixes and street abbreviations seen in its Source-2/3 records; région ↔ département variants from the lexicon.

Synthetic records go through exactly the same normalise → block → featurise path. Their blocking recall (0.960–0.961) is close to real India (0.937). This suggests comparable difficulty; it is not proof of realism.

**Validation (leave-India-out):** US labels only 0.9098 → US + synthetic India **0.9193** → real India labels 0.9513. Singleton false merges drop from 24.1 % to 14.1 %.

**Assumptions:** France uses the same operator families at similar rates. Operators unique to France that never appear in the data cannot be generated.

## Isolation of test-derived information
Test files are used only as **unlabelled inputs**:
- France lexicon from identical-name pseudo-pairs;
- synthetic seeds and vocabulary.

No test labels exist or are used. The lexicon and model fitting use TRAIN entities only, all selection uses VALID, and TEST is scored once with the frozen design; the submission models are then refit on all labelled entities.

Experiments tried and rejected are listed in [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md).
