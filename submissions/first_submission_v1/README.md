# FIRST SUBMISSION — v1

- `matching_results.tsv`: the leaderboard file. Official validator PASS (`--check-ids`); 1,732,544 rows, one per test Source-1 entity; 5,682,608 matched ids; no duplicate rows, ids or columns.
- `matching_results.zip`: the same file compressed (40.7 MB). Only for use if the portal explicitly accepts .zip.
- Model: Doppelgänger, two-stage LightGBM + ownership + expected-F0.5 lists; France scored by the synthetic-data model.
- Holdout (train split, 220,929 entities): macro F0.5 0.9674, precision 0.9912, recall 0.9310.
- Test-set score: unknown until the leaderboard evaluates it.
