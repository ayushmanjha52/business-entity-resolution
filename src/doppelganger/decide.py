"""From calibrated pair probabilities to match lists.

1. owner():  every Source-2/3 record belongs to at most one Source-1 entity (true for 100% of the 7.6M
             labelled pairs) -> keep only the pair where the record's probability is highest.
2. select(): the metric is F0.5 *per entity*, so each entity's list is the prefix (by probability) that
             maximises expected F0.5:  E[F | top-k] ~ 1.25*sum_topk(p) / (k + 0.25*sum(p)),
             versus the empty list, which scores 1 only if the entity has no match: prod(1 - p).
"""
import numpy as np
import pandas as pd


def owner(s1, c, p) -> np.ndarray:
    """Boolean mask: pair is the highest-probability claim on its record (ties -> first)."""
    o = np.lexsort((-p, c))
    first = np.r_[True, c[o][1:] != c[o][:-1]]
    keep = np.zeros(len(c), bool)
    keep[o[first]] = True
    return keep


def select(s1, p, min_prob=0.0, beta2=0.25) -> np.ndarray:
    """Boolean mask of selected pairs (pairs should already be owner-filtered)."""
    o = np.lexsort((-p, s1))
    s, q = s1[o], p[o].astype(np.float64)       # running sums over millions of pairs: float32 loses the per-entity part
    start = np.r_[True, s[1:] != s[:-1]]
    gid = np.cumsum(start) - 1
    first = np.flatnonzero(start)
    k = np.arange(len(s)) - first[gid] + 1
    cs = np.cumsum(q)
    top = cs - np.r_[0, cs][first][gid]                                   # sum of top-k within entity
    G = np.bincount(gid, weights=q)[gid]
    ef = (1 + beta2) * top / (k + beta2 * G)
    ef0 = np.exp(np.bincount(gid, weights=np.log1p(-np.minimum(q, 1 - 1e-7))))  # P(no true match among candidates)
    best = pd.Series(ef).groupby(gid).idxmax().to_numpy()                 # row of the best prefix per entity
    kbest = np.where(ef[best] > ef0, k[best], 0)
    sel = (k <= kbest[gid]) & (q >= min_prob)
    mask = np.zeros(len(s1), bool)
    mask[o[sel]] = True
    return mask


def cross_source_fill(s1, c, p, sel, src3, min_prob) -> np.ndarray:
    """Add the best other-source record to entities whose list holds a single source.

    80 % of labelled entities are matched in both Source 2 and Source 3, only 14 % in exactly one, so a list
    with one source usually misses a record. The best still-unassigned record of the missing source joins the
    list when its probability is at least min_prob (0.5 chosen on VALID: +0.0004 on VALID and TEST)."""
    rec_src3 = src3[c]
    lists = pd.DataFrame({"s": s1[sel], "s3": rec_src3[sel]}).groupby("s").s3.agg(["min", "max"])
    single = lists[lists["min"] == lists["max"]]
    missing_s3 = pd.Series(~single["max"].astype(bool), index=single.index)
    taken = np.zeros(c.max() + 1, bool)
    taken[c[sel]] = True
    m = np.isin(s1, single.index) & ~taken[c] & (p >= min_prob)
    m &= rec_src3 == missing_s3.reindex(s1).fillna(False).to_numpy(bool)
    out = sel.copy()
    if m.any():
        d = pd.DataFrame({"i": np.flatnonzero(m), "s": s1[m], "c": c[m], "p": p[m]}).sort_values("p", ascending=False)
        out[d.drop_duplicates("c").drop_duplicates("s").i.to_numpy()] = True    # one per entity, each record once
    return out


def threshold(s1, c, p, t) -> np.ndarray:
    """Baseline decision: independent global threshold (no ownership, no per-entity optimisation)."""
    return p >= t
