# 60-second demo

```bash
python -m doppelganger explore        # → http://127.0.0.1:8765
```
(Requires `artifacts/evidence.parquet`, built by `python -m doppelganger evidence`.)

| Time | Click | What the audience sees | What to say |
|---|---|---|---|
| 0–20 s | **Show me a suspicious look-alike** | A Source-1 business, plus several records with nearly the same name at the same street. One or more *look-alikes* have a high **similarity model** bar (stage 1, often > 0.9) and a near-zero **Doppelgänger** bar. They are marked **REJECTED**, and the ground-truth badge says *different business (decoy)*. The house numbers are highlighted: green where they agree with Source 1, red where they are shifted (e.g. 668 vs 661). | "Similarity says these are the same company. They're not: the generator plants sibling businesses with the house number moved a few doors down. Doppelgänger compares each record with its rivals and the number geometry, and rejects the impostor." |
| 20–40 s | **Looks nothing alike — still the same business** | Matched records whose names are pseudo-words (`Onyxonyx`), URLs (`orthopedicsafehealth.com`), native script (`बेस्ट इंफोटेक लिमिटेड`) or aliases (`Koraviaria formerly known as …`). Every one is marked **MATCHED** and ground truth agrees. | "The least similar records are the safest matches. They're recognisable by the noise operator that produced them, so the model trusts the address." |
| 40–50 s | **Same name, different branch** | A candidate with an identical name that is **CLAIMED BY ANOTHER ENTITY**, i.e. another branch of the same chain. | "Every record belongs to at most one business. Here the stronger claim wins, so the branch isn't double-counted." |
| 50–60 s | **France (learned without labels)** + scroll to *How much does the Doppelgänger logic buy?* | Test entities from France (no labels), then the holdout table: fuzzy baseline → final system, and the leave-India-out table. | "France never appears in training. We re-ran the reverse-engineered noise generator on unlabelled French records to create training pairs. On India, treated as unlabelled, this halved false merges on singletons." |

Every candidate card has **Why this score?**, which shows per-feature SHAP contributions from the stage-2 model.

Tips:
- Each button cycles through different cases (click again).
- The search box accepts a business name fragment or an `S1-…` id from the evidence store: the 10 % validation holdout with ground truth, plus a 60k test sample.
