# Candidate generation (blocking)

Code: `src/doppelganger/blocking.py`. Settings: `[blocking]` in `configs/default.toml`.

## Key families
Keys are computed identically for Source 1 and Source 2/3, per country. Each key is packed into one `int64` as `family « 58 | a « 29 | b`. Token ids index a per-country dictionary built from the Source-1 tokens; numbers are capped below 2^29.

| Family | a × b | Why |
|---|---|---|
| NAME×NUM | core-name token × one of the first 3 address numbers | survives name reordering and typos in other tokens; separates decoys (different number) |
| ADDR×NUM | street/locality token × number | the only route for pseudo-names, native-script names not in the dictionary, and URL names |
| NAME×NAME | unordered pair of core-name tokens (first 5) | records whose address is empty or has no number |
| NAME×ADDR | core-name token (first 3) × street/locality token (first 6) | same-name branches when numbers are missing (Source 1 has many identical names at different addresses) |
| NOSPACE | whole core name without spaces | `orthopedicsafehealth.com` → `orthopedicsafehealth` |
| NAME1 (v3) | one core-name token | a typo in the other token of a two-word name, with an empty address (`Cowgill Cble LLC`) |
| ADDR×ADDR (v3) | unordered pair of street/locality tokens | pseudo-names whose house number was rewritten |
| ADDR×NUM, digit-deleted (v3) | Source-1 primary number with one digit deleted × street token | `677 WINDSTAR PL` vs `2677 Windstar Place`: 45 % of numbered true pairs whose primary number is missing are single-digit deletions |

Normalisation (v3) also feeds blocking: digits inside alphanumeric name tokens are read as letters (`c0mmerce` → `commerce`, `6cs` → `gcs`), and URL names are segmented into Source-1 vocabulary words (`impexprivate.com` → `impex`).

## Index and query (no Python loops over records)
1. Source-1 keys are sorted; `np.unique` gives each distinct key's start offset and count. Keys shared by more than `s1_key_cap = 150` Source-1 records are dropped (v2: 30; the median non-selective key that hid a true match was shared by 58 records).
2. Source-2/3 records are processed one million at a time. `np.searchsorted` locates their keys, and matching ranges are expanded with `np.repeat` arithmetic.
3. Votes: each shared key adds `1 / log2(1 + n)`, where n is the number of Source-1 records sharing it. Votes are summed with `np.unique(..., return_inverse)` + `np.bincount`.
4. Per record: keep entities ranked in the top `top_per_record = 8` **and** with votes ≥ `min_rel_votes = 0.35` × the record's best. Each record belongs to at most one entity, so ranking from the record side keeps the set small without losing the right entity.

## Measured quality, v3 (train split, all 2.2M entities)
| | v2 | v3 |
|---|---|---|
| train pairs | 18,503,975 | 18,577,959 |
| pair recall | 0.9605 | **0.9794** (best entity only: 0.9555) |
| oracle macro F0.5 | 0.9846 | **0.9929** |
| test pairs | 19,756,139 | 20,700,123 |
| runtime train / test | 240 s / 223 s | 2,408 s / 1,770 s (chunk 250k, more keys) |

Diagnosis that drove v3 (302,073 missed v2 pairs): 29 % shared no key at all (URL concatenations, typos with empty address, dropped digits), 49 % shared only keys held by more than 30 Source-1 records, 22 % shared a selective key but lost the per-record ranking. Raising the cap alone gave recall 0.9710; the new families, homoglyph folding and URL segmentation add the rest.

## Measured quality, v2 (train split, all 2.2M entities)
Recall/cost curve for the final key set (from `python -m doppelganger block`):

| Cut | Pairs | Pair recall |
|---|---|---|
| best entity only (rank < 1) | 10,153,446 | 0.9341 |
| rank < 2 | 20,132,667 | 0.9471 |
| rank < 4 | 39,810,678 | 0.9557 |
| rank < 8 | 78,202,020 | 0.9613 |
| rel ≥ 0.2 | 26,219,369 | 0.9612 |
| **rel ≥ 0.35 (used)** | **18,503,975** | **0.9605** |
| rel ≥ 0.5 | 15,487,772 | 0.9589 |

- Without the relative cut (top-8), the oracle macro F0.5 (a perfect matcher on these candidates) is 0.985: US 0.993, India 0.973. The relative cut removes 0.0008 recall.
- Test split: 19,756,139 pairs, i.e. 11.4 per Source-1 entity and 1.98 per Source-2/3 record. The reduction ratio against all cross pairs is 0.99999886. 3,970 of 1,732,544 test entities have no candidate.
- Runtime: 240 s (train, 10.3M records) and 223 s (test, 10.0M records). The largest index (US train) holds 31.8M keys.

## History
Version 1 (4 families, raw vote counts, top-6) reached pair recall 0.904. Diagnosis showed that half of the missed records received no candidate and half of them had no house number. The NAME×ADDR family and rarity weighting raised recall to 0.961. See `docs/EXPERIMENT_LOG.md`.

## Remaining misses
Most remaining misses are records with an empty or number-less address whose core name is shared by more than 30 Source-1 entities (common generated names such as "national fellowship"). Those keys are dropped as non-selective, and even when a key survives, the name alone cannot tell the branches apart.
