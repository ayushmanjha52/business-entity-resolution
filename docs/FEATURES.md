# Features

Code: `src/doppelganger/features.py`. Pairwise features are computed in chunks of 1.5M pairs: rapidfuzz `cpdist` (element-wise, all cores) for strings and numpy broadcasting for numbers. They are cached in `artifacts/feats_{split}.parquet`, row-aligned with `artifacts/pairs_{split}.parquet`.

## Pairwise (37, used by both stages)
| Group | Feature | Meaning |
|---|---|---|
| Name | `nm_tset` | token-set ratio of core names (legal forms removed) |
| | `nm_tsort` | token-sort ratio of full names (legal form included) |
| | `nm_ratio_ns`, `nm_partial_ns`, `nm_jw` | ratio / partial ratio / Jaro-Winkler on the space-free core (catches URL names and concatenations) |
| | `tok_diff`, `n_tok1` | extra or missing name words (decoys add a descriptor); Source-1 name length |
| Address | `ad_tset` | token-set ratio of alphabetic address tokens after lexicon mapping and abbreviation canonicalisation |
| | `ad_tsort` | token-sort ratio of the full normalised address |
| | `loc_overlap` | number of shared localities (states, régions, big cities) |
| House-number geometry | `num_p_in` | Source 1's primary number appears among the record's numbers |
| | `num_p_first` | …and it is the record's first number |
| | `num_frac1` | share of Source 1's numbers kept in the record |
| | `num_frac2` | share of the record's numbers that are Source 1's (injected door numbers lower it) |
| | `num_logdiff` | log(1 + min \|record number − primary number\|): **the decoy signature** is a small non-zero shift |
| | `num_drop` | record number equals the primary with a leading or trailing digit dropped (215 → 15) |
| | `n_nums1`, `n_nums2` | number counts on each side |
| Noise flags (record) | `c_f_pseudo`, `c_f_alias`, `c_f_url`, `c_f_script`, `c_f_empty_addr`, `c_f_junk` | which generator operator produced the record |
| Other | `c_s3`, `votes` | Source-3 indicator; rarity-weighted blocking votes |
| Rarity (v3) | `nm_idf_cov1`, `nm_idf_cov2` | share of each side's name-token rarity mass (IDF over Source 1) found on the other side |
| | `nm_idf_rare` | IDF of the rarest shared name token (a shared rare word is strong evidence; a shared "services" is not) |
| | `ad_idf_cov1`, `ad_idf_cov2`, `ad_idf_rare` | the same for street/locality tokens |
| Name frequency (v3) | `s1_name_n`, `c_name_n` | log count of Source-1 entities with this exact core name (Source 1's, the record's); identical names shared by dozens of branches make name similarity uninformative |
| | `nm_exact` | identical core names |
| Numbers (v3) | `num_del_any` | the record carries Source 1's primary number with any one digit deleted (2677 → 677) |
| | `num_long` | both sides carry a ≥ 5-digit number (PIN-like) and one is shared (NaN when either has none) |
| Rarity (v3) | `nm_idf_cov1`, `nm_idf_cov2` | share of each side's name-token rarity mass (IDF over Source 1) found on the other side |
| | `nm_idf_rare` | IDF of the rarest shared name token (a shared rare word is strong evidence; a shared "services" is not) |
| | `ad_idf_cov1`, `ad_idf_cov2`, `ad_idf_rare` | the same for street/locality tokens |
| Name frequency (v3) | `s1_name_n`, `c_name_n` | log count of Source-1 entities with this exact core name (Source 1's, the record's); identical names shared by dozens of branches make name similarity uninformative |
| | `nm_exact` | identical core names |
| Numbers (v3) | `num_del_any` | the record carries Source 1's primary number with any one digit deleted (2677 → 677) |
| | `num_long` | both sides carry a ≥ 5-digit number (PIN-like) and one is shared (NaN when either has none) |

The house-number features are undefined (NaN) when either side has no number. LightGBM learns a separate path for missing values.

## Competition (9, stage 2 only)
Built from out-of-fold stage-1 probabilities `p1`:

| Feature | Rivals considered | Meaning |
|---|---|---|
| `e_rank`, `e_rel`, `e_margin` | other records of the same entity | rank, ratio to the best, margin to the next best |
| `e_n`, `e_sum` | same | candidate count; expected number of matches (Σ p1) |
| `e_sib` | same | look-alikes around the entity: name ≥ 90 but primary number absent |
| `c_rank`, `c_n`, `c_margin` | other entities proposed for the same record | rank; competitor count; margin over the strongest rival entity |

## Group coherence (4, stage 2 only, `model.coherence = true`)
Each record is compared with its entity's most confident *other* record (the "anchor", chosen by stage-1 score):

| Feature | Meaning |
|---|---|
| `coh_p` | the anchor's stage-1 probability (how reliable the second view is) |
| `coh_nm`, `coh_ad` | token-set similarity of the record to the anchor: name / address |
| `coh_num` | the record and the anchor share an address number (NaN if either has none) |

A record that is badly corrupted relative to Source 1 (empty address, pseudo-name, dropped number) often still resembles a sibling record that is a sure match.

Measured on the holdout: F0.5 0.9674 → **0.9678**. Recall rises from 0.9310 to 0.9322 with precision unchanged (0.9912), and every slice improves (singletons 0.9518 → 0.9546). `coh_p` becomes the top stage-2 feature by gain.

## What the features are *for*
Neither the sibling decoys nor the disguised true records can be separated by similarity alone:
- the **number geometry** and **rival margins** reject look-alikes;
- the **noise flags** and **space-free name** features keep disguised true records.

## v3 effect
With the v3 features, blocking and a larger model, VALID macro F0.5 went from 0.9678 to 0.9806. The four new feature groups enter the stage-2 top 12 by gain: `c_name_n` (1st), `nm_idf_rare`, `ad_idf_rare`, `s1_name_n`. The gains of blocking, features and model size were not ablated separately in v3.

## v3 effect
With the v3 features, blocking and a larger model, VALID macro F0.5 went from 0.9678 to 0.9806. The four new feature groups enter the stage-2 top 12 by gain: `c_name_n` (1st), `nm_idf_rare`, `ad_idf_rare`, `s1_name_n`. The gains of blocking, features and model size were not ablated separately in v3.

## Importance and ablation (v2) (v2)
Stage-2 gain (percent), final labels-only model: see `python -m doppelganger train` log, "stage-2 top features".

In the first full model, the leading features were:
- `nm_ratio_ns` 43 %
- `e_sum` 14 %
- `ad_tset` 13 %
- `num_logdiff` 10 %
- `e_n` 5 %
- `votes` 3 %
- `c_margin` 2 %

Removing the 7 lowest-gain noise flags reduced holdout F0.5 from 0.9674 to 0.9667, so all features were kept (`model.exclude = []`).

Country is **not** a feature. This lets the models apply to countries without labels (France).
