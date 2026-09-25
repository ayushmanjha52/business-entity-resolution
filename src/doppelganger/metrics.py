"""Challenge metric (macro F0.5 over Source-1 entities, singletons included) and slice reports."""
import numpy as np
import pandas as pd


def pair_code(s1, c, n_cand):
    return np.asarray(s1, np.int64) * n_cand + np.asarray(c, np.int64)


def per_entity(pred, gt, entities, n_cand):
    """pred/gt: (s1_rows, cand_rows). Returns DataFrame indexed by entity with tp, n_pred, n_gt, P, R, F."""
    ent = pd.Index(entities)
    pc_, gc = pair_code(*pred, n_cand), pair_code(*gt, n_cand)
    tp_mask = np.isin(pc_, gc)
    def count(s1, w=None):
        pos = ent.get_indexer(s1)
        ok = pos >= 0
        return np.bincount(pos[ok], weights=None if w is None else w[ok], minlength=len(ent))
    d = pd.DataFrame({"tp": count(pred[0], tp_mask.astype(float)), "n_pred": count(pred[0]), "n_gt": count(gt[0])},
                     index=ent)
    empty = (d.n_pred == 0) & (d.n_gt == 0)
    d["F"] = np.where(empty, 1.0, 1.25 * d.tp / np.maximum(d.n_pred + 0.25 * d.n_gt, 1e-9))
    d["P"] = np.where(d.n_pred > 0, d.tp / np.maximum(d.n_pred, 1), np.where(d.n_gt == 0, 1.0, np.nan))
    d["R"] = np.where(d.n_gt > 0, d.tp / np.maximum(d.n_gt, 1), np.nan)
    return d


def summary(d: pd.DataFrame) -> dict:
    single = d.n_gt == 0
    return {"entities": len(d), "F0.5": d.F.mean(), "precision": d.P.mean(), "recall": d.R.mean(),
            "singleton_FP_rate": float((d.n_pred[single] > 0).mean()) if single.any() else float("nan"),
            "coverage": float((d.n_pred > 0).mean()), "pairs_pred": int(d.n_pred.sum()), "pairs_tp": int(d.tp.sum())}


def report(d: pd.DataFrame, slices: dict[str, pd.Series]) -> pd.DataFrame:
    """Summary overall and for each boolean/categorical slice (aligned to d.index)."""
    rows = {"ALL": summary(d)}
    for name, s in slices.items():
        s = s.reindex(d.index)
        for v in pd.unique(s.dropna()):
            if s.dtype == bool and not v:
                continue
            rows[f"{name}={v}" if s.dtype != bool else name] = summary(d[s == v])
    return pd.DataFrame(rows).T


def _auc(y, p):
    """ROC-AUC via the rank-sum statistic (ties averaged)."""
    r = pd.Series(p).rank().to_numpy()
    pos = y == 1
    n1, n0 = pos.sum(), (~pos).sum()
    return float((r[pos].sum() - n1 * (n1 + 1) / 2) / max(n1 * n0, 1))


def _average_precision(y, p):
    o = np.argsort(-p, kind="stable")
    tp = np.cumsum(y[o] == 1)
    prec = tp / np.arange(1, len(o) + 1)
    return float((prec * (y[o] == 1)).sum() / max(tp[-1] if len(tp) else 0, 1))


def pair_report(y, sel, p, n_gt) -> dict:
    """Pair-level confusion matrix over candidate pairs. Recall counts true pairs lost in blocking as misses."""
    y, sel = np.asarray(y) == 1, np.asarray(sel, bool)
    tp, fp = int((sel & y).sum()), int((sel & ~y).sum())
    fn_cand, tn = int((~sel & y).sum()), int((~sel & ~y).sum())
    fn_block = int(n_gt - y.sum())
    prec = tp / max(tp + fp, 1)
    rec = tp / max(n_gt, 1)
    return {"candidate_pairs": int(len(y)), "true_pairs": int(n_gt), "true_pairs_in_candidates": int(y.sum()),
            "blocking_recall": float(y.sum() / max(n_gt, 1)), "predicted_pairs": tp + fp,
            "TP": tp, "FP": fp, "FN": fn_cand + fn_block, "FN_in_candidates": fn_cand, "FN_blocking": fn_block, "TN": tn,
            "accuracy_candidate_pairs": (tp + tn) / max(len(y), 1), "precision": prec, "recall": rec,
            "F1": 2 * prec * rec / max(prec + rec, 1e-12), "F0.5_micro": 1.25 * prec * rec / max(0.25 * prec + rec, 1e-12),
            "ROC_AUC": _auc(y.astype(int), np.asarray(p)), "PR_AUC": _average_precision(y.astype(int), np.asarray(p))}


def threshold_table(y, p, n_gt, grid=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)) -> pd.DataFrame:
    """Pair precision / recall / F1 / F0.5 of a plain global threshold on the pair probability."""
    y = np.asarray(y) == 1
    rows = {}
    for t in grid:
        s = p >= t
        tp = (s & y).sum()
        pr, rc = tp / max(s.sum(), 1), tp / max(n_gt, 1)
        rows[t] = {"pairs": int(s.sum()), "precision": pr, "recall": rc, "F1": 2 * pr * rc / max(pr + rc, 1e-12),
                   "F0.5_micro": 1.25 * pr * rc / max(0.25 * pr + rc, 1e-12)}
    return pd.DataFrame(rows).T.rename_axis("threshold")
