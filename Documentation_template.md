# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** bluebeyond  
**Team Members:** Ayushman Jha  
**Submission Date:** 2026-09-25

---

## 1. Executive Summary
**Doppelgänger** is a CPU-only blocking + two-stage LightGBM pipeline built around one measured property of this data: the hard negatives are **sibling look-alikes** (same name and street, house number shifted a few units), while the least similar records (pseudo-names, aliases, URLs, native script, empty addresses) are almost always **true** matches.
- Records are judged against their **rivals** (competition features), and each Source-2/3 record belongs to **at most one** entity.
- Each entity's list **maximises expected F0.5** from calibrated probabilities.
- France has no labels, so we **re-applied the reverse-engineered noise generator** to unlabelled French records to create training data.

Result: macro F0.5 **0.9752** on 220,804 held-out TEST entities scored once after the design was frozen, **at test-like decoy density**. The same entities at plain train density: 0.9783. The previous model scored 0.9552 at test-like density.

---

## 2. Methodology

### 2.1 Problem Analysis
Measured on the training data:
- Only 5.6 % of Source-1 entities are singletons; the mean is 3.46 matches (max 11). All 7,638,365 matched records belong to exactly one entity, and every match is within one country.
- The 26 % of unmatched Source-2/3 records are mostly **decoys**. Source 1's first house number appears in 83 % of US true matches but in only 4 % of US look-alike decoys (India: 90 % vs 34 %).
- Noise operators, with the share of flagged records that are true matches:
  - pseudo-word names 1.7 % of records (98 % true);
  - alias markers such as "X formerly known as Y" 1.3 % (100 % true);
  - URL/handle names 4.3 % (96 %);
  - native-script names 7.3 % (73 %);
  - empty addresses 3.4 % (98 %).

  Also present: typos and homoglyphs, legal-suffix moves, injected door numbers (`#7-67`, `H.no 954`, `N°`), number variants (1681D, 1681-1683), `null`/`N/A`, and component reordering.
- State and région representations swap between sources: India full name ↔ code (TN) ↔ native script (தமிழ்நாடு); US code ↔ full; France région ↔ département (Hauts-de-France ↔ Nord).
- 31 % of Source-1 businesses share a street with another business, and Source 1 contains many identical names at different addresses (branches).
- Test differs from train: 15 % France (unseen), more India, and ~5.7 Source-2/3 records per entity vs 4.7 in train.

- **The test split is decoy-heavier than train**, measured without labels on the candidate sets. Records per entity: 5.75 vs 4.68. Hard same-street siblings per entity: US 1.04 → 1.76, India 0.64 → 1.04. True matches per entity are unchanged. A model validated at train density over-states the test score: 0.9806 at train density vs 0.9552 at test-like density.

### 2.2 Solution Strategy
**Approach Type:** Hybrid — learned-lexicon normalisation, multi-key blocking, two-stage gradient boosting with competition features, constrained per-entity decision, and self-supervised synthetic data for the unlabelled country.  
**Core Innovation:** Training and validating at the test split's measured decoy density (`train_aug`: synthetic sibling decoys thinned per country to the label-free hard-sibling gap). On test-like VALID this lifts macro F0.5 from 0.9552 to 0.9754. Also: Judging each candidate by the noise operator that produced it and by its rivals, not just by its similarity to the reference. For France, **generator inversion** creates labelled training pairs without any labels, validated by a leave-India-out experiment.

---

## 3. Candidate Generation (Blocking)
- **Blocking keys used**, integer-packed and country-scoped:
  - name token × address number;
  - street/locality token × number, also with the Source-1 number minus one digit (`2677` → `677`);
  - unordered name-token pair;
  - name token × street/locality token;
  - whole space-free name;
  - single name token;
  - unordered street/locality token pair.

  Before keys are built, digits inside words are read as letters (`c0mmerce`) and URL names are segmented into Source-1 words (`impexprivate.com` → `impex`). Keys shared by more than 150 Source-1 records are dropped. Votes are rarity-weighted (1/log2(1+n)). Each Source-2/3 record keeps its top-8 entities with at least 35 % of its best vote.
- **Candidate pairs generated:** train 18,577,959; test 20,700,123 (11.9 per Source-1 entity, 2.08 per Source-2/3 record; reduction ratio 0.99999880). Duplicate pairs cannot occur: votes are aggregated per (record, entity).
- **How you ensured true matches were not lost:**
  - We diagnosed misses from version 1 (pair recall 0.904). Half the missed records got no candidate, and half of those had no house number.
  - We added the name × street family to separate same-name branches.
  - We measured the recall/cost curve before choosing cut-offs.
  - Version 3 diagnosed the 302,073 pairs version 2 still missed. 29 % shared no key at all (URL concatenations, typos with empty address, dropped digits). 49 % shared only keys held by more than 30 Source-1 records. 22 % were ranked out.

  Final pair recall is **0.9794** (version 2: 0.9605), and the oracle macro F0.5 ceiling is **0.9929** (0.9846).

---

## 4. Matching Model

**Features used** (37 pairwise + 9 competition + 4 group-coherence):
- **Name features:**
  - token-set ratio on the core name (legal forms removed) and token-sort on the full name;
  - ratio, partial ratio and Jaro-Winkler on the space-free name (URL names);
  - word-count difference;
  - exact core-name match, and how many Source-1 entities share this exact name (either side's name); identical generated names shared by dozens of branches make name similarity uninformative;
  - rarity (IDF) weighted token overlap in both directions and the rarest shared token.

  Native-script names are transliterated with a 1,312-token dictionary learned from the training pairs (93 % coverage).
- **Address features:**
  - token-set and token-sort ratios after learned state/région equivalences and street-abbreviation canonicalisation;
  - locality overlap;
  - **house-number geometry:** primary number present / first, share of numbers kept both ways, log of the smallest number shift (the decoy signature), dropped-digit relations (leading/trailing and any single digit), agreement of long PIN-like numbers, number counts;
  - rarity-weighted street/locality token overlap.
- **Other:**
  - noise-operator flags (pseudo-name, alias, URL, native script, empty address, junk prefix), Source-3 indicator, blocking votes;
  - **competition features** from out-of-fold stage-1 scores: rank, relative score and margin among the entity's candidates; the entity's candidate count, expected match count and look-alike count; rank, count and margin among the entities competing for the same record.
  - **group-coherence features**: the record compared (name, address, shared number) with the entity's most confident other record, plus that record's confidence.

**Model type:** LightGBM (MIT), two stages (a third stage was tried and was not robust), trained on `train_aug` (real pairs + calibrated synthetic sibling decoys), 127 leaves, learning rate 0.06, 600 trees. Chosen on VALID pairs against logistic regression (ROC-AUC 0.9768), random forest (0.9973), ExtraTrees (0.9964) and other LightGBM sizes (0.9987–0.9988). Stage 1 uses pairwise features with 2-fold out-of-fold scores by entity; stage 2 uses pairwise + competition features. There are two model sets: real labels for US/India, and real labels + synthetic France for France. Development models train on TRAIN entities (50 %); the submission models are refit on every labelled entity once the design is frozen.

**Threshold selection method:** No global threshold.
1. Each Source-2/3 record goes to its highest-probability entity.
2. Each entity's list is the probability-sorted prefix that maximises expected F0.5, `1.25·Σtop-k p / (k + 0.25·Σp)`, against `Π(1−p)` for the empty list.

Probabilities are calibrated, within about 0.01–0.03 per bin. The only free parameter, a probability floor under this rule, was searched on VALID (0–0.5) and frozen at 0.5; it is nearly inert (0.98052 → 0.98056). It beats the best plain global threshold (0.9803 at 0.7).

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro):** **0.9752** on 220,804 TEST entities at test-like density (US 0.9794, India 0.9688). VALID (selection set, test-like): 0.9754. Previous model at test-like density: 0.9552. Tuned fuzzy baseline: 0.6529.
- **Pair level (TEST, test-like):** precision 0.9921, recall 0.9462 (blocking misses counted), F1 0.9686, ROC-AUC 0.9986, PR-AUC 0.9969. TP 722,171, FP 5,758, TN 1,391,647, FN 41,054 (25,211 in candidates + 15,843 lost in blocking).
- **Leakage control:** entity-level split; the lexicon is learned from TRAIN-role labels only (fixed in this version); stage-2 inputs use out-of-fold stage-1 scores; the decision floor is frozen on VALID; TEST is scored once. Unlabelled-country test (India treated as unlabelled): 0.9098 with US labels only, **0.9193** adding synthetic data, 0.9513 with real labels.
- **Common false positives (wrong merges):**
  - look-alike decoys whose shifted number is not visible (India addresses carry several numbers): singleton false-merge rate 2.5 % (was 4.5 %);
  - records whose true entity was never a candidate: blocking missed it, so the record went to the best look-alike.
- **Common false negatives (missed matches):**
  - empty-address records: 43 % of in-candidate misses, against 3.4 % of true pairs (slice recall 0.84);
  - records lost in blocking (pair recall 0.9794);
  - records claimed by a rival entity with a higher score (5,275 on VALID).
- **Why not 0.99:** a perfect matcher on these candidates would score 0.9929. The rest of the gap (0.012) is mostly name-only records of businesses whose generated name is shared by many Source-1 entities, where the data does not identify the right branch.

---

## 6. Conclusion
The data was generated by a noise process whose negatives are look-alikes and whose hardest positives look least alike. Modelling the operators and the competition between candidates, rather than tuning a similarity threshold, lifts macro F0.5 from 0.65 to 0.975 on held-out entities at the test split's decoy density.

Everything runs on a 16 GB CPU machine: largest measured peak 12.9 GB (final refit); full runtime in the README benchmark. Every normalisation resource is learned from the provided files, and the unlabelled country is handled with validated self-supervision.

---

## Appendix

### A. Code Artefacts
`code/business_entity_resolution/`:
- `src/doppelganger/`: `normalize`, `lexicon`, `blocking`, `features`, `model`, `decide`, `metrics`, `synth`, `selfsup`, `pipeline`, `evidence`, `explorer/`;
- `configs/default.toml`: every setting;
- `tests/`: unit tests, explorer API test, end-to-end test;
- `docs/`: research report, methodology, blocking, features, model, validation, experiment log, demo.

Entry point:
```
pip install -r requirements.txt
set PYTHONPATH=src            # export PYTHONPATH=src on Linux/macOS
python -m doppelganger all    # prepare → block → train → validate → predict → evidence
```
This writes `output/matching_results.tsv` and `output/candidate_pairs.tsv`. Individual stages: `prepare`, `block`, `train`, `validate`, `predict`, `loco`, `evidence`, `explore`.

### B. Additional Results
See `docs/VALIDATION.md` (slices by country, singletons, decoy exposure, unusual names, empty addresses, candidate-set size; ablations) and `docs/EXPERIMENT_LOG.md` (every experiment, including rejected ones).

The explorer (`python -m doppelganger explore`) shows, for any held-out or test entity, each candidate's decision, its ground truth where available, its stage-1 vs final score, plain-language evidence and SHAP contributions.

---

**Note:** No external data, APIs, geocoding or pretrained language models are used. All libraries are MIT/BSD/Apache-licensed.
