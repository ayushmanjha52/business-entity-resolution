# Experiment log

Chronological. Every number was measured on the real data; the log files are in `artifacts/run_*.log`.
Unless noted, scores are macro F0.5 on the 10 % Source-1 validation holdout (220,929 entities, of which 88,360 are India). From #19 on, this holdout is called VALID and a separate 10 % TEST split is scored once.

| # | Change | Measurement | Decision |
|---|---|---|---|
| 1 | Base normalisation cache (pyarrow) | 22.3M records normalised; e.g. train S2 (5.0M rows) in 56 s | kept |
| 1a | Bug: Indic vowel signs treated as separators (`राम` → `र म`) | fixed by keeping `\p{M}` in token classes | fixed |
| 2 | Lexicon v1 | 278 component equivalences; street lines such as `31`→`unit 31` included | rejected |
| 2a | Lexicon v2: components containing digits excluded | 211 equivalences (France 9, India 48); 1,312 transliteration tokens covering 93 % of native-script name tokens | kept |
| 3 | Blocking v1: 4 key families, raw votes, top-6 per record | pair recall 0.904 (US 0.932 / India 0.862); oracle F0.5 0.952; test 52.7M pairs | rejected |
| 3a | Diagnosis of misses | 49 % of missed records got no candidate at all; 47 % of missed records have no house number; Source 1 contains many identical names at different addresses | — |
| 3b | Blocking v2: + name×street keys, rarity-weighted votes, top-8 | pair recall 0.9613; oracle F0.5 0.985 (US 0.993 / India 0.973); 78.2M train pairs | kept |
| 3c | v2 + relative-vote cut (≥35 % of the record's best) | 18.5M train pairs, recall 0.9605 (curve: rank<1 0.934, rel≥0.2 0.9612, rel≥0.5 0.9589); test 19.8M pairs | **kept** |
| 4 | Pair features | 18.5M pairs in 125 s (26 features) | kept |
| 5 | First full model, no synthetic data | B0 fuzzy 0.7239 · B1 stage-1+threshold 0.9614 · B2 +decision 0.9621 · B3a stage-2+0.5 0.9658 · **B3 final 0.9674** | kept |
| 5a | Calibration check (holdout) | predicted vs observed within about 0.01 in every probability bin | justifies expected-F0.5 |
| 5b | Exact Monte-Carlo expected F0.5 vs closed form | 0.9676 vs 0.9677; 11.6 s vs 6.3 s | closed form kept |
| 5c | Probability floor 0.05 vs none | 0.9674 vs 0.9677 | floor removed |
| 6 | First test submission | official validator PASS with `--check-ids`; empty lists: France 5.2 %, India 6.5 %, US 5.6 % | superseded |
| 7 | Leave-India-out, A: US labels only | India F0.5 0.9098 (P 0.946, R 0.884, singleton false-merge rate 24.1 %) | — |
| 7a | Synth rate fix: address-typo rate measured as near-miss words | 0.798 → 0.113 | fixed |
| 7b | Leave-India-out, B: US + synthetic India (701k records, 26 % decoys, synthetic blocking recall 0.960) | India F0.5 **0.9193** (P 0.969, R 0.862, singleton false-merge rate 14.1 %) | synthetic block kept |
| 7c | Leave-India-out, C: real India labels | India F0.5 0.9513: upper bound | — |
| 8 | Ablation: remove 7 low-gain noise flags from the model | 0.9674 → 0.9667 | flags kept |
| 9 | Single model trained with synthetic France (830k records, 37 % decoys) | holdout (US/India) 0.9665, i.e. −0.0009 vs #5 | replaced by #10 |
| 10 | **Per-country routing**: labelled countries use the real-label model, unlabelled (France) the synthetic-augmented model | holdout 0.9674 (labelled countries unchanged); test: 16,426,133 pairs → real-label model, 3,330,006 (France) → synthetic model | kept |
| 11 | From-scratch reproducibility run (`all --config configs/repro.toml`, fresh work dir) | pair recall 0.9605 and holdout F0.5 0.9674 / P 0.9912 / R 0.9310, with identical pair counts (716,847 predicted, 711,457 correct) | reproducible |
| 12 | Explorer explanations checked on real cases | fixed misleading text: "look-alike signature" shown on a true match; "wins over look-alikes" shown on a rejected decoy; now concrete number differences ("502 absent; record has 52 instead") | fixed |
| 13 | Submission size analysis (portal upload stalled at 95.7 MB) | minimum valid file (one row per test entity, all lists empty) is 24.1 MB; zip/gzip 40.7 MB; 0 duplicates or extra columns in the file | a valid file under 10 MB is impossible in the required format |
| 14 | Group-coherence features (record vs the entity's most confident other record), labels-only model | holdout 0.9674 → **0.9678** (P 0.9912 = 0.9912, R 0.9310 → 0.9322; US 0.9786, India 0.9518, singletons 0.9518 → 0.9546); `coh_p` top stage-2 feature | **kept (v2)** |

| 15 | Loss decomposition of v2 on VALID | blocking 0.0154, false merges 0.0074, in-candidate misses ≈ 0.006 (0.9678 total) | blocking first |
| 16 | Diagnosis of 302,073 missed train pairs | no shared key 29 %, only keys shared by > 30 S1 records 49 % (median 58), selective key but ranked out 22 %; 45 % of numbered misses are single-digit deletions | — |
| 17 | Blocking: key cap 30 → 150 (old normalisation) | 18,967,629 pairs, recall 0.9710, oracle 0.9903 | kept |
| 18 | Leak found: lexicon sampled labelled pairs of **all** entities incl. the validation holdout | lexicon now learned from TRAIN-role pairs only; full rebuild | fixed |
| 19 | Split: TRAIN 50 % / VALID 10 % (same entities as the old holdout) / TEST 10 % (never evaluated before) | — | kept |
| 20 | v3 normalisation (homoglyph folding, URL segmentation) + NAME1 / ADDR×ADDR / digit-deleted keys, cap 150 | 18,577,959 train pairs, recall **0.9794**, oracle **0.9929**; test 20,700,123 pairs | **kept** |
| 21 | v3 features (rarity overlap, name frequency, exact name, any-digit deletion, long numbers) + LightGBM 127 leaves / 600 trees | VALID **0.9806** (P 0.9949, R 0.9548); `c_name_n` top stage-2 feature | **kept** |
| 22 | Decision floor grid on VALID (0–0.5) | 0.98052 … 0.98056 at 0.5 | frozen at 0.5 |
| 23 | Model comparison on VALID pairs (LogReg, RF, ExtraTrees, 3 LightGBM sizes) | ROC-AUC 0.9768 / 0.9973 / 0.9964 / 0.9987–0.9988 | LightGBM 127 leaves |
| 24 | One-time TEST evaluation | macro F0.5 **0.9806** (P 0.9953, R 0.9539, ROC-AUC 0.9992) | matches VALID |

| 25 | Leaderboard feedback: v3 file scored ~0.970 against 0.9806 on US/India TEST | the gap is not France alone (see #30) | — |
| 26 | French-aware normalisation (Allée/Cours/Route…, French function words, floor noise) | first version lost 0.0008 on US/India VALID: stop words `n` (canonical "North"), `d`, `l` (initials, block letters) | those three removed; rest kept; US/India VALID back to 0.9806 (two-stage) |
| 27 | Decision layer: prefix sums in float32 over millions of pairs | owned pairs with p ≈ 0.8 dropped from lists; fixed with float64: +0.0001 | **fixed** |
| 28 | Third stage (rivals from out-of-fold stage-2 scores) | +0.0009 (experiment), −0.0005 (pipeline run) | not robust: `model.stages = 2` |
| 29 | Self-training for unlabelled countries, India treated as unlabelled | India VALID 0.9392 → **0.9424** (singleton false merges 8.8 % → 7.9 %) | **kept** for France |
| 30 | Label-free comparison of candidate sets | test has 5.75 records/entity vs 4.68; hard same-street siblings/entity US 1.04 → 1.76, India 0.64 → 1.04; true matches/entity unchanged | test is decoy-heavier |
| 31 | Stress test: plain model on VALID + synthetic decoys at the raw +1.07/entity gap | 0.9806 → 0.9466 (synthetic decoys are all hard siblings: too many) | calibrate instead |
| 32 | Decoys thinned per country to the measured hard-sibling gap (US keep 93.8 %, India 52.3 %) | plain model on test-like VALID: **0.9552** (P 0.9645, singleton false merges 11.3 %) | test-like VALID adopted |
| 33 | **Train on test-like train_aug** | test-like VALID 0.9552 → **0.9754** (P 0.9916); plain VALID 0.9806 → 0.9783 | **kept (v4)** |
| 34 | One-time TEST, test-like (frozen v4 design) | macro F0.5 **0.9752** (P 0.9921, R 0.9462, ROC-AUC 0.9986) | matches VALID |

| 35 | Stage-2 variants on test-like VALID (v4): 255 leaves / lr 0.04 / 1000 trees; min_child 20 / 800 trees; bag of 3 seeds; average of variants | 0.9751; 0.9741; 0.9756; 0.9755 vs 0.9754 | no change: all within noise |

| 36 | Remaining blocking misses (80k sampled of 157,614) | 2 % share no key; raising the cap to 300 / 500 / 1000 recovers 6.3 % / 9.5 % / 12.8 %; the rest share a selective key but rank below the top 8 (recall: top-6 0.9777, top-8 0.9794) | wider blocking judged ~+0.001: not run |

| 37 | Leaderboard: v4 file 0.9709 (v3 ~0.970) | the test-like gain (+0.020 locally) did not show up: real test siblings are easier than the synthetic ones | — |
| 38 | France label-free check: numberless look-alikes (true 99.5 % in US) predicted only 75.7 % in France | suspected a France recall leak | investigated (#39) |
| 39 | France scored by A = French model, B = labelled US/India model, C = mean | matches/entity 3.359 / 3.358 / 3.334; numberless look-alikes predicted 75.7 % / 72.7 % / 74.6 % | the models agree: no leak; file unchanged (`synth.unlabelled_weight` = 1.0) |

## Negative or neutral results (kept for honesty)
- Stage-2 gain importance ranks the noise flags near zero, but removing them costs 0.0007 F0.5. Gain importance under-reports features that act in rare but decisive cases.
- The synthetic block closes only about 23 % of the gap to real labels in the leave-India-out test (0.9098 → 0.9193 vs 0.9513). Operators specific to the unlabelled country cannot be generated if they are never observed.
- Adding the synthetic block to the shared model costs the labelled countries 0.0009. This is why the final system routes by country.
- LightGBM with 255 leaves / 800 trees scored best F1 0.9799 vs 0.9798 for 127 leaves at twice the fit time: not adopted.
- The v2 decision-floor ablation (0.05 vs none) and the v3 grid agree: the floor barely matters once probabilities are calibrated.
- A third stage looked like +0.0009 in isolation but lost 0.0005 in the next full run: within noise, so the simpler two-stage model stays.
- Plain-density validation over-stated test performance by about 0.025 because the test candidate sets are more crowded with hard siblings; every v4 number is test-like.
