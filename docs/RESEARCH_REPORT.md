> **Data analysis: current. Scores: v2 (0.9678), superseded** — current results are in [VALIDATION.md](VALIDATION.md).

# Research Report — Phases 1–5 (pre-implementation)

> Status: written before any pipeline code. It records what the data showed and the plan at the time. What was actually built and measured is in METHODOLOGY.md, BLOCKING.md, FEATURES.md, MODEL.md, VALIDATION.md and EXPERIMENT_LOG.md. Main deviations from the plan: blocking gained a name×street key family and weighted votes (EXPERIMENT_LOG #3); the synthetic block is used through per-country model routing (#10); "shadow-registry" (concept 8) was not built.

All numbers below were measured on the provided `student_resource/` data (scripts run in a scratch area; no project code written yet).

## Phase 1 — What is actually in Student Resource

| File | Rows | Notes |
|---|---|---|
| `train_source1.tsv` | 2,206,821 | US 1.32M, India 0.88M. No empty fields, no duplicate IDs |
| `train_source2.tsv` | 5,034,616 | 3.4% empty address |
| `train_source3.tsv` | 5,285,603 | 3.3% empty address |
| `train_ground_truth.tsv` | 2,206,821 | one row per S1; 7,638,365 positive pairs |
| `test_source1.tsv` | 1,732,544 | **India 47%, US 38%, France 15%** (train: India 40%, no France) |
| `test_source2/3.tsv` | 4.89M / 5.08M | |
| `README.md`, `Documentation_template.md`, `utils/validate_submission.py` | – | problem spec, write-up template, stdlib format validator |

Schema is 4 string columns (`entity_id, business_name, business_address, country`). There are **no timestamps, users, behaviour logs, images, models or APIs** — temporal/behavioural/personalisation angles do not exist in this data and are not pursued.

Hardware: 16 GB RAM, 16 logical CPUs, no CUDA GPU → everything must be CPU-first and streaming-friendly (~10M candidate records per split).

### Structural facts from the ground truth
- Singletons: **5.6%** only. Matches per S1: mode 3, mean 3.46, max 11; both S2 and S3 contribute (S2 and S3 each contain internal duplicates of the same entity).
- **Every S2/S3 record matches at most one S1** (0 of 7.64M matched IDs repeat) → a hard mutual-exclusion constraint.
- Country agreement between matched records: **100%** → blocking within country is lossless.
- No leakage: entity-ID numbers and row order are uncorrelated with matches (r ≈ 0.002).
- 26% of S2/S3 records match nothing.

### The data is produced by a generator with identifiable noise operators
| Operator (observed) | Example | Rate | P(true match \| operator) |
|---|---|---|---|
| Name replaced by a synthetic pseudo-word | `Onyxonyx`, `Iriwexbelo`, `Fluxfluxcira` | 1.7% | **0.98** |
| Alias / rebrand marker | `Koraviaria formerly known as General Electronics Partners`, `f/k/a`, `t/a`, `DBA` | 1.3% | **1.00** |
| Name rendered as a domain/handle | `orthopedicsafehealth.com`, `@jexfirst` | 4.3% | 0.96 |
| Native-script transliteration | `बेस्ट इंफोटेक लिमिटेड`; states in Tamil/Gujarati/Devanagari | 7.3% | 0.73 |
| Empty address | – | 3.4% | **0.98** |
| Junk prefix / brackets | `>> `, `-- `, `[LLC]`, `(Techinfra)` | 8% | ~0.75 |
| Char typos, homoglyphs, accents | `Ttuaesb`, `KEYST0NE`, `Fócus` | common | – |
| Legal-suffix moves/swaps, honorifics, person-name inversion | `Pvt. X Ltd.`, `Sri …`, `Fonseca, Wilma` | common | – |
| Injected door/plot number | `#7-67`, `H.no 954`, `Plot 803`, `N°` | common | – |
| Number variants on true matches | `1681D`, `1681-1683`, `26259-B`, `052`→`52` | common | – |
| Region/state representation swap | India: full ↔ code (`Tamil Nadu`↔`TN`); US: code ↔ full; **France: région ↔ département** (`Hauts-de-France`↔`Nord`) | common | – |
| Placeholders | `null`, `N/A`, `PO Box …`, `FCDP` | ~3.5% | – |

Takeaway: the records that look *least* like their S1 entity are almost always true matches. The generator uses them to hurt recall, not precision.

### The precision trap: sibling decoys
Unmatched S2/S3 records are mostly **near-clones of a real business**: same or near-same name (often + a descriptor such as `Holdings`, `North`, `Infratech`, or `LLC LLC`), same street and city, **house number shifted by a small amount** (239 vs 238, 4975 vs 4962, 668 vs 661, `46-A` vs `35-A`, `B-23/B` vs `B-18/B`).

| Measured on 1.5k pairs per group | US true | US decoy | India true | India decoy |
|---|---|---|---|---|
| S1's first address number present in candidate | 0.83 | **0.04** | 0.90 | 0.34 |
| Any number shared | 0.88 | 0.17 | 0.95 | 0.57 |
| Name token-set similarity (median) | 100 | 90 | 93 | 81 |

- These siblings are almost never themselves in S1 (only 0.13% of S1 share a de-numbered street and first two name tokens with another S1). They are "shadow" businesses present only in S2/S3.
- Separately, **31% of S1 records share a street (numbers removed) with another S1 business**, and some share an identical address (three different companies at `45 A, Block Eu Pitampura`). Address-only matching is therefore unsafe unless the name is recognised as synthetic.
- Test has **~5.7 S2/S3 records per S1 vs 4.7 in train**. If the number of true matches per entity is similar, test has roughly **2× the decoy density**, so a threshold tuned on raw train will over-merge.

## Phase 2 — Concepts derived from these signals
1. **Sibling-aware matcher** — every candidate is scored relative to its competitors (same-S1 rivals, and rival S1s for the same record), with a number-geometry feature (exact / suffix / range / leading-zero / small-shift / different). Enabled by the decoy finding. Hard to copy: requires discovering that the negatives are adversarial siblings.
2. **Mutual-exclusion global assignment** — use the "≤1 S1 per record" constraint to resolve S1 businesses that share an address.
3. **Noise-operator fingerprinting** — detect which operator produced each record and route it: pseudo-word names → address-only evidence; alias → compare the post-marker name; domain → compare concatenated tokens; transliteration → script-aware comparison.
4. **Self-learned gazetteer, zero external data** — learn `state ↔ code ↔ native-script` and France `région ↔ département` equivalences from co-occurrence in high-confidence pairs. France is learned **unsupervised from the unlabeled test set**.
5. **Generator inversion for an unseen country** — re-apply the reverse-engineered noise operators to France S1 records to synthesise labelled French pairs and decoys, giving self-supervised domain adaptation.
6. **Expected-F0.5 decision layer** — scoring is macro per entity, so choose, per S1, the subset that maximises *expected* F0.5 from calibrated probabilities instead of a global threshold. This handles singletons and ambiguous entities naturally.
7. **Decoy-density calibration** — re-weight validation to test's candidate density before tuning decisions.
8. **Shadow-registry discovery** — cluster unmatched S2/S3 records to surface businesses missing from the master (S1). *Hypothesis; not yet verified.*
9. **Evidence dossier / explainer** — per S1 entity, show accepted records, rejected siblings and the specific evidence (e.g. "number 668 ≠ 661") behind each decision.
10. **Transliteration aligner** — learn a Devanagari↔Latin character mapping from train pairs so Hindi-script names compare in Latin space, with no external transliteration library.
11. **Ambiguity risk map** — per-entity false-merge risk from sibling density, usable by data stewards as a review queue.

## Phase 3 — The unexpected connection
**Name noise type + address number geometry + mutual exclusion + candidate density → a record's identity is decided by its rivals, not by its similarity to the reference.**
Plain similarity says a decoy ("Lockport Advanced Telefonica Corp, 668 Locust St") is a 0.91 match. Only comparing it with the true sibling ("…Inc, 661 Locust Street") and the other candidates reveals the trap. Conversely, the least similar records (pseudo-word names, empty addresses, aliases) are the safest matches. Similarity and truth are anti-correlated at both ends, which a standard similarity-plus-classifier pipeline gets wrong.

## Phase 4 — Selected direction: **Doppelgänger**
An entity-resolution engine that is explicitly built to catch business look-alikes. It combines concepts 1, 2, 3, 4, 6 and 7, has an explainer UI (concept 9), and uses **generator inversion for France** (concept 5) as its "nobody expected this" layer.
Why it emerged from the data: the precision-heavy metric plus adversarial sibling decoys plus an unlabeled third country are the three real difficulties in this dataset, and each component addresses one measured difficulty. It is not a dashboard. The leaderboard file is produced by the same engine that the UI explains.

## Phase 5 — Architecture (planned)
```
student_resource/dataset ──► ingest (TSV→parquet, normalise, operator tags, parsed address: street key, number set, locality, region)
                          ──► gazetteer learner (train pairs + unsupervised test co-occurrence)
                          ──► blocking (country-scoped; keys: street-key+number, rare name tokens, char-ngram TF-IDF top-k within locality; capped)
                                 └─► candidate_pairs.tsv  (+ recall ceiling / reduction ratio report)
                          ──► features (name, address, number geometry, operator flags, rival/competition features)
                          ──► LightGBM (MIT) pairwise model; France via generator-inverted synthetic pairs
                          ──► decision: mutual-exclusion assignment → per-entity expected-F0.5 subset
                                 └─► matching_results.tsv (validated with utils/validate_submission.py)
                          ──► evidence store (parquet) ──► FastAPI + single-page explorer ("why matched / why rejected")
```
- **Validation:** hold out 10% of S1 entities *with* all their S2/S3 records plus a proportional share of decoys; report macro F0.5 per country, at train density and at simulated test density.
- **Licensing:** pandas, scikit-learn, LightGBM, RapidFuzz, FastAPI (BSD/MIT). No LLM required, so far below the 8B limit. No external lookups (gazetteer is learned from the provided files only).
- **Privacy/security:** local-only explorer, read-only access to the data, input validation on IDs; some business names contain personal names, so nothing is published externally.
- **Scale:** processed per country in chunks; test Source 2+3 is about 10M records, sized for 16 GB RAM.

Open items to measure before relying on them: blocking recall ceiling; whether orphan records cluster (concept 8); the size of the France synthetic-data gain.
