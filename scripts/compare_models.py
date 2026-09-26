"""Pair-classifier comparison on the stage-1 (pairwise) feature store.

    python scripts/compare_models.py [--train-pairs 1500000] [--config configs/default.toml]

Every model is fitted on pairs of TRAIN-role entities and scored on pairs of VALID-role entities (the split is
by Source-1 entity, so no business appears on both sides). TEST entities are never touched.
Writes <work_dir>/validation/model_comparison.csv.
"""
import argparse
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from doppelganger import config, model, pipeline  # noqa: E402
from doppelganger.utils import log  # noqa: E402


def best_f1(y, p):
    o = np.argsort(-p)
    tp = np.cumsum(y[o])
    f1 = 2 * tp / (np.arange(1, len(o) + 1) + y.sum())
    i = int(f1.argmax())
    return float(f1[i]), float(p[o][i])


def other_boosters(cfg) -> dict:
    """XGBoost / CatBoost (Apache-2.0) at a capacity comparable to the LightGBM setting; skipped if not installed."""
    out = {}
    try:
        import xgboost as xgb
        out["XGBoost (lossguide, 127 leaves, lr 0.06, 600 trees)"] = lambda: xgb.XGBClassifier(
            n_estimators=600, learning_rate=0.06, max_depth=0, max_leaves=127, grow_policy="lossguide",
            tree_method="hist", subsample=0.7, colsample_bytree=0.8, min_child_weight=1, n_jobs=-1,
            random_state=cfg.seed)
    except ImportError:
        pass
    try:
        from catboost import CatBoostClassifier
        out["CatBoost (depth 8, lr 0.08, 1000 trees)"] = lambda: CatBoostClassifier(
            iterations=1000, learning_rate=0.08, depth=8, thread_count=-1, random_seed=cfg.seed, verbose=0,
            allow_writing_files=False)
    except ImportError:
        pass
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--train-pairs", type=int, default=1_500_000)
    a = ap.parse_args()
    cfg = config.load(a.config) if a.config else config.load()
    rng = np.random.default_rng(cfg.seed)

    s1, cand = pipeline.load(cfg, "train", ["entity_id"])
    pairs = pipeline.candidates(cfg, "train")
    ps, pcand = pairs["s1"].to_numpy(), pairs["c"].to_numpy()
    y, _ = pipeline.labels(cfg, s1["entity_id"], cand["entity_id"], ps, pcand)
    role = pipeline.split_roles(cfg, s1["entity_id"].to_numpy(zero_copy_only=False))[ps]
    X = pipeline.model_view(cfg, pipeline.pair_features(cfg, "train"))
    tr = np.flatnonzero(role == pipeline.TRAIN)
    tr = rng.choice(tr, min(a.train_pairs, len(tr)), replace=False)
    va = np.flatnonzero(role == pipeline.VALID)
    Xtr, ytr, Xva, yva = X.iloc[tr], y[tr], X.iloc[va], y[va]
    log(f"model comparison: {len(tr):,} TRAIN pairs, {len(va):,} VALID pairs, {X.shape[1]} features")

    impute = SimpleImputer(strategy="constant", fill_value=-1.0)
    candidates = {
        "LogisticRegression (standardised, C=1)": make_pipeline(impute, StandardScaler(), LogisticRegression(max_iter=300)),
        "RandomForest (200 trees)": make_pipeline(impute, RandomForestClassifier(
            200, min_samples_leaf=20, n_jobs=-1, random_state=cfg.seed)),
        "ExtraTrees (200 trees)": make_pipeline(impute, ExtraTreesClassifier(
            200, min_samples_leaf=20, n_jobs=-1, random_state=cfg.seed)),
    }
    rows = {}
    for name, clf in candidates.items():
        t = time.time()
        sub = rng.choice(len(tr), min(500_000, len(tr)), replace=False) if "Forest" in name or "Trees" in name else slice(None)
        clf.fit(Xtr.iloc[sub], ytr[sub])
        rows[name] = (clf.predict_proba(Xva)[:, 1], time.time() - t)
        log(f"  {name}: {rows[name][1]:.0f}s")
    for leaves, lr, trees in ((63, 0.08, 400), (cfg.model.num_leaves, cfg.model.learning_rate, cfg.model.n_estimators),
                              (255, 0.05, 800)):
        t = time.time()
        params = {**model.params(cfg), "num_leaves": leaves, "learning_rate": lr}
        b = lgb.train(params, lgb.Dataset(Xtr, ytr), num_boost_round=trees)
        name = f"LightGBM ({leaves} leaves, lr {lr}, {trees} trees)"
        rows[name] = (b.predict(Xva), time.time() - t)
        log(f"  {name}: {rows[name][1]:.0f}s")
    for name, make in other_boosters(cfg).items():
        t = time.time()
        clf = make()
        clf.fit(Xtr, ytr)
        rows[name] = (clf.predict_proba(Xva)[:, 1], time.time() - t)
        log(f"  {name}: {rows[name][1]:.0f}s")

    table = pd.DataFrame({name: {"ROC_AUC": roc_auc_score(yva, p), "PR_AUC": average_precision_score(yva, p),
                                 "log_loss": log_loss(yva, np.clip(p, 1e-6, 1 - 1e-6)),
                                 "best_F1": best_f1(yva, p)[0], "best_F1_threshold": best_f1(yva, p)[1],
                                 "fit_s": round(sec)} for name, (p, sec) in rows.items()}).T
    out = cfg.paths.work_dir / "validation"
    out.mkdir(exist_ok=True)
    table.to_csv(out / "model_comparison.csv")
    pd.set_option("display.width", 200)
    log("pair classifiers on VALID pairs (stage-1 features):\n" + table.round(4).to_string())


if __name__ == "__main__":
    main()
