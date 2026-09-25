"""Pair features (chunked, vectorised) and competition features.

Pairwise:  name/address string similarity (rapidfuzz cpdist, multi-threaded), address-number geometry,
           locality agreement, and the noise-operator flags that explain *why* a true match can look unlike
           its entity (pseudo-name, alias, URL, native script, empty address).
Competition: every pair is re-described relative to its rivals — the other records competing for the same
           Source-1 entity and the other Source-1 entities competing for the same record.
"""
import numpy as np
import pandas as pd
import pyarrow.compute as pc
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler
from rapidfuzz.process import cpdist

from .blocking import digit_deletions

STRING_FEATURES = [  # (name, scorer, column)
    ("nm_tset", fuzz.token_set_ratio, "name_core"),
    ("nm_tsort", fuzz.token_sort_ratio, "name_full"),
    ("nm_ratio_ns", fuzz.ratio, "name_nospace"),
    ("nm_partial_ns", fuzz.partial_ratio, "name_nospace"),
    ("nm_jw", JaroWinkler.normalized_similarity, "name_nospace"),
    ("ad_tset", fuzz.token_set_ratio, "addr_alpha"),
    ("ad_tsort", fuzz.token_sort_ratio, "addr_norm"),
]
FLAGS = ["f_alias", "f_url", "f_junk", "f_script", "f_empty_addr", "f_pseudo"]
PAIR_COLS = ["name_core", "name_full", "name_nospace", "addr_alpha", "addr_norm", "localities", "n_name_tok",
             "n_nums", "entity_id", "country", "name_tokens", "addr_tokens"] + FLAGS
TOK_W = 6    # tokens per record compared in the rarity-weighted overlap features


def _token_ids(lst, dictionary, width=TOK_W):
    lst = lst.combine_chunks() if hasattr(lst, "combine_chunks") else lst
    ids = pc.fill_null(pc.index_in(lst.flatten(), value_set=dictionary), -1).to_numpy().astype(np.int32)
    off = lst.offsets.to_numpy()
    lens = np.diff(off)
    m = np.full((len(lst), width), -1, np.int32)
    for j in range(width):
        has = lens > j
        m[has, j] = ids[off[:-1][has] + j]
    return m


class TokenRarity:
    """Inverse document frequency of Source-1 name / address tokens and core-name frequency per country."""

    def __init__(self, s1):
        self.idf = {}
        for col in ("name_tokens", "addr_tokens"):
            vc = pc.value_counts(s1[col].combine_chunks().flatten())
            self.idf[col] = (vc.field("values"), np.log(s1.num_rows / vc.field("counts").to_numpy()).astype(np.float32))
        vc = pc.value_counts(pc.binary_join_element_wise(s1["country"], s1["name_core"], "|"))
        self.name_keys, self.name_counts = vc.field("values"), vc.field("counts").to_numpy()

    def overlap(self, col, t1, t2, prefix):
        """Share of each side's rarity mass found on the other side, and the rarest shared token."""
        d, idf = self.idf[col]
        a, b = _token_ids(t1[col], d), _token_ids(t2[col], d)
        wa = np.where(a >= 0, idf[np.maximum(a, 0)], 0).astype(np.float32)
        wb = np.where(b >= 0, idf[np.maximum(b, 0)], 0).astype(np.float32)
        nb = pc.list_value_length(t2[col].combine_chunks()).to_numpy()
        wb_unknown = np.clip(np.minimum(nb, TOK_W) - (b >= 0).sum(1), 0, None) * idf.max()
        eq = (a[:, :, None] == b[:, None, :]) & (a[:, :, None] >= 0)
        sa, sb = eq.any(2), eq.any(1)
        tot_a, tot_b = wa.sum(1), wb.sum(1) + wb_unknown
        nan = np.float32(np.nan)
        return {f"{prefix}_cov1": np.where(tot_a > 0, (wa * sa).sum(1) / np.maximum(tot_a, 1e-6), nan).astype(np.float32),
                f"{prefix}_cov2": np.where(tot_b > 0, (wb * sb).sum(1) / np.maximum(tot_b, 1e-6), nan).astype(np.float32),
                f"{prefix}_rare": (wa * sa).max(1).astype(np.float32)}

    def name_count(self, t):
        """Number of Source-1 records (same country) whose core name equals this record's."""
        key = pc.binary_join_element_wise(t["country"], t["name_core"], "|")
        i = pc.fill_null(pc.index_in(key, value_set=self.name_keys), -1).to_numpy().astype(np.int64)
        return np.where(i >= 0, self.name_counts[np.maximum(i, 0)], 0).astype(np.float32)


def number_features(n1: np.ndarray, n2: np.ndarray) -> dict:
    """Address-number geometry between S1 numbers n1 (m,k) and candidate numbers n2 (m,k); -1 = absent.

    Decoys keep the street but shift the house number by a small amount; true variants keep the number
    and add noise around it (suffix letters, ranges, injected door numbers, dropped digits)."""
    p = n1[:, :1]
    v1, v2 = n1 >= 0, n2 >= 0
    eq = (n1[:, :, None] == n2[:, None, :]) & v1[:, :, None] & v2[:, None, :]
    in2, in1 = eq.any(2), eq.any(1)
    has_p, has_c = v1[:, 0], v2.any(1)
    diff = np.where(v2 & (p >= 0), np.abs(n2.astype(np.int64) - p), np.iinfo(np.int64).max).min(1)
    nd = lambda x: np.floor(np.log10(np.maximum(x, 1))).astype(np.int64) + 1
    # dropped leading / trailing digit(s) of the primary number (e.g. 215 -> 15, 052 -> 52 already folded)
    pp = np.broadcast_to(p, n2.shape).astype(np.int64)
    drop = v2 & (pp >= 10) & (n2 != pp) & ((n2 == pp % 10 ** nd(n2)) | (n2 == pp // 10 ** np.maximum(nd(pp) - nd(n2), 0)))
    dels = digit_deletions(p[:, 0])                                          # S1 primary with one digit deleted
    del_any = ((n2[:, :, None] == dels[:, None, :]) & v2[:, :, None] & (dels[:, None, :] >= 0)).any((1, 2))
    long1, long2 = v1 & (n1 >= 10_000), v2 & (n2 >= 10_000)
    long_eq = (eq & long1[:, :, None]).any((1, 2))
    nan = np.float32(np.nan)
    return {
        "num_p_in": np.where(has_p & has_c, in2[:, 0], nan).astype(np.float32),
        "num_p_first": np.where(has_p & has_c, n2[:, 0] == p[:, 0], nan).astype(np.float32),
        "num_frac1": np.where(has_p & has_c, in2.sum(1) / np.maximum(v1.sum(1), 1), nan).astype(np.float32),
        "num_frac2": np.where(has_p & has_c, in1.sum(1) / np.maximum(v2.sum(1), 1), nan).astype(np.float32),
        "num_logdiff": np.where(has_p & has_c, np.log1p(np.minimum(diff, 10 ** 9)), nan).astype(np.float32),
        "num_drop": np.where(has_p & has_c, (drop.any(1) & ~in2[:, 0]), nan).astype(np.float32),
        "n_nums1": v1.sum(1).astype(np.float32), "n_nums2": v2.sum(1).astype(np.float32),
        "num_del_any": np.where(has_p & has_c, del_any & ~in2[:, 0], nan).astype(np.float32),
        "num_long": np.where(long1.any(1) & long2.any(1), long_eq, nan).astype(np.float32),
    }


def _loc_features(l1, l2):
    ov = np.fromiter((len(set(a) & set(b)) if a is not None and b is not None and len(a) and len(b) else -1
                      for a, b in zip(l1, l2)), np.float32, len(l1))
    return {"loc_overlap": np.where(ov < 0, np.nan, ov).astype(np.float32)}


def pair_features(s1, cand, ps, pcand, votes, cfg, chunk=1_500_000) -> pd.DataFrame:
    k = cfg.prep.max_numbers
    s1n = np.stack([s1[f"num{j}"].to_numpy() for j in range(k)], 1)
    cn = np.stack([cand[f"num{j}"].to_numpy() for j in range(k)], 1)
    s1c, cc = s1.select(PAIR_COLS), cand.select(PAIR_COLS)
    rar = TokenRarity(s1c)
    s1_name_n, c_name_n = rar.name_count(s1c), rar.name_count(cc)
    parts = []
    for i in range(0, len(ps), chunk):
        a, b = ps[i:i + chunk], pcand[i:i + chunk]
        t1, t2 = s1c.take(a), cc.take(b)
        f = {name: cpdist(t1[col].to_numpy(zero_copy_only=False), t2[col].to_numpy(zero_copy_only=False),
                          scorer=sc, workers=-1).astype(np.float32) for name, sc, col in STRING_FEATURES}
        f.update(number_features(s1n[a], cn[b]))
        f.update(_loc_features(t1["localities"].to_numpy(zero_copy_only=False), t2["localities"].to_numpy(zero_copy_only=False)))
        for fl in FLAGS:
            f[f"c_{fl}"] = t2[fl].to_numpy(zero_copy_only=False).astype(np.float32)
        f["tok_diff"] = (t2["n_name_tok"].to_numpy() - t1["n_name_tok"].to_numpy()).astype(np.float32)
        f["n_tok1"] = t1["n_name_tok"].to_numpy().astype(np.float32)
        f["c_s3"] = pc.starts_with(t2["entity_id"], "S3").to_numpy(zero_copy_only=False).astype(np.float32)
        f["votes"] = votes[i:i + chunk].astype(np.float32)
        f.update(rar.overlap("name_tokens", t1, t2, "nm_idf"))
        f.update(rar.overlap("addr_tokens", t1, t2, "ad_idf"))
        f["nm_exact"] = pc.equal(t1["name_core"], t2["name_core"]).to_numpy(zero_copy_only=False).astype(np.float32)
        f["s1_name_n"], f["c_name_n"] = np.log1p(s1_name_n[a]), np.log1p(c_name_n[b])
        parts.append(pd.DataFrame(f))
    return pd.concat(parts, ignore_index=True)


def competition_features(s1: np.ndarray, c: np.ndarray, score: np.ndarray, sib: np.ndarray) -> pd.DataFrame:
    """Relative position of each pair among its rivals, from a first-stage score.

    e_* : rivals = other candidate records of the same Source-1 entity
    c_* : rivals = other Source-1 entities proposed for the same candidate record
    sib : pair-level 'look-alike' indicator (similar name, number disagreement) counted per entity."""
    d = pd.DataFrame({"s": s1, "c": c, "p": score, "sib": sib})
    ge, gc = d.groupby("s").p, d.groupby("c").p
    e_max, c_max = ge.transform("max"), gc.transform("max")
    out = pd.DataFrame({
        "e_rank": ge.rank(ascending=False, method="min"),
        "e_rel": d.p / np.maximum(e_max, 1e-6),
        "e_n": ge.transform("size"),
        "e_sum": ge.transform("sum"),
        "e_sib": d.groupby("s").sib.transform("sum"),
        "c_rank": gc.rank(ascending=False, method="min"),
        "c_n": gc.transform("size"),
        # margin to the best *other* entity for this record (negative = another entity wins the record)
        "c_margin": d.p - np.where(d.p >= c_max, _second(d, "c"), c_max),
        "e_margin": d.p - np.where(d.p >= e_max, _second(d, "s"), e_max),
    })
    return out.astype(np.float32)


COHERENCE_COLS = ["name_core", "addr_alpha", "num0", "num1", "num2"]


def coherence_features(s1, c, p1, cand, chunk=2_000_000) -> pd.DataFrame:
    """Compare each record with the entity's most confident *other* record (its "anchor").

    Records of one business resemble each other even when one of them is badly corrupted relative to the
    Source-1 reference (empty address, pseudo-name, dropped number): the anchor supplies that second view."""
    o = np.lexsort((-p1, s1))
    s = s1[o]
    start = np.r_[True, s[1:] != s[:-1]]
    first = np.flatnonzero(start)
    gid = np.cumsum(start) - 1
    size = np.diff(np.r_[first, len(s)])
    top = o[first][gid]
    second = np.where(size[gid] > 1, o[np.minimum(first + 1, len(s) - 1)][gid], -1)
    anc = np.empty(len(s1), np.int64)
    anc[o] = np.where(o == top, second, top)                 # pair index of the anchor, -1 if none
    has = np.flatnonzero(anc >= 0)
    t = cand.select(COHERENCE_COLS)
    nums = np.stack([t[f"num{j}"].to_numpy() for j in range(3)], 1)
    nan = np.full(len(s1), np.nan, np.float32)
    out = {"coh_p": nan.copy(), "coh_nm": nan.copy(), "coh_ad": nan.copy(), "coh_num": nan.copy()}
    out["coh_p"][has] = p1[anc[has]]
    for i in range(0, len(has), chunk):
        h = has[i:i + chunk]
        a, b = c[anc[h]], c[h]
        ta, tb = t.take(a), t.take(b)
        for key, col in (("coh_nm", "name_core"), ("coh_ad", "addr_alpha")):
            out[key][h] = cpdist(ta[col].to_numpy(zero_copy_only=False), tb[col].to_numpy(zero_copy_only=False),
                                 scorer=fuzz.token_set_ratio, workers=-1)
        na, nb = nums[a], nums[b]
        shared = ((na[:, :, None] == nb[:, None, :]) & (na[:, :, None] >= 0)).any((1, 2))
        out["coh_num"][h] = np.where((na >= 0).any(1) & (nb >= 0).any(1), shared, np.nan)
    return pd.DataFrame(out)


def _second(d, key):
    """Second-largest score within each group (0 when the group has one member), aligned to rows."""
    o = d.sort_values([key, "p"], ascending=[True, False])
    second = o[o[key].duplicated()].groupby(key).p.first()
    return d[key].map(second).fillna(0.0).to_numpy()
