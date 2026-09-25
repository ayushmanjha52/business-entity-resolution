# SUBMISSION v3

- `matching_results.tsv`: leaderboard file. Official validator PASS (`--check-ids`); 1,732,544 rows (1,630,468 with matches); 5,772,346 matched ids; 96.8 MB; SHA-256 ded50416…0113.
- Changes vs v2: blocking v3 (pair recall 0.9605 → 0.9794), homoglyph folding and URL segmentation, rarity / name-frequency / number features, LightGBM 127 leaves × 600 trees, leak-free lexicon, models refit on all labelled entities.
- Held-out TEST (220,804 train entities never used for any choice): macro F0.5 0.9806 (precision 0.9953, recall 0.9539). VALID: 0.9806. v2 on the same VALID entities: 0.9678.
- Test predictions per country: France 858,687 matches (6.0 % empty lists), India 2,665,742 (6.0 %), US 2,247,917 (5.7 %).
- Leaderboard score: unknown until the portal evaluates it.
