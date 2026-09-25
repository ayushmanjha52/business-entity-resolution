# SUBMISSION — v2

- `matching_results.tsv`: leaderboard file. Official validator PASS (`--check-ids`); 1,732,544 rows (1,627,199 with matches); 5,690,496 matched ids; 95.8 MB; SHA-256 ab89c363…5eed.
- Change vs v1: stage-2 models gain 4 group-coherence features (each record compared with its entity's most confident other record). The France model is retrained the same way.
- Holdout (220,929 train entities): macro F0.5 0.9678 (v1: 0.9674); precision 0.9912; recall 0.9322 (v1: 0.9310).
- Test predictions per country: France 861,984 matches (5.9 % empty lists), India 2,593,219 (6.5 %), US 2,235,293 (5.6 %).
- Test-set score: unknown until the leaderboard evaluates it.
