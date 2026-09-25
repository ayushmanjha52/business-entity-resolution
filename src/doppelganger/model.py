"""LightGBM (MIT licence) pair classifiers: stage 1 (pairwise evidence) and stage 2 (+ competition)."""
import lightgbm as lgb
import numpy as np
import pandas as pd

from .utils import log


def params(cfg, seed_offset=0):
    m = cfg.model
    return dict(objective="binary", learning_rate=m.learning_rate, num_leaves=m.num_leaves,
                min_child_samples=m.min_child_samples, subsample=m.subsample, subsample_freq=1,
                colsample_bytree=m.colsample_bytree, n_jobs=-1, verbose=-1, seed=cfg.seed + seed_offset)


def fit(X: pd.DataFrame, y: np.ndarray, cfg, seed_offset=0) -> lgb.Booster:
    ds = lgb.Dataset(X, y, free_raw_data=True)
    b = lgb.train(params(cfg, seed_offset), ds, num_boost_round=cfg.model.n_estimators)
    log(f"  trained on {len(y):,} pairs ({y.mean():.1%} positive), {X.shape[1]} features")
    return b


def oof(X, y, groups, cfg) -> np.ndarray:
    """Out-of-fold predictions, folds split by Source-1 entity so no entity scores its own training pairs."""
    fold = (pd.util.hash_array(np.asarray(groups, np.int64)) % cfg.model.folds).astype(int)
    out = np.zeros(len(y), np.float32)
    for f in range(cfg.model.folds):
        tr = fold != f
        out[~tr] = fit(X[tr], y[tr], cfg, f).predict(X[~tr])
    return out


def importance(b: lgb.Booster) -> pd.Series:
    return pd.Series(b.feature_importance("gain"), index=b.feature_name()).sort_values(ascending=False)
