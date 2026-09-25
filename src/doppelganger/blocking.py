"""Candidate generation with integer-coded blocking keys and a sorted inverted index.

Keys (all country-scoped, built identically for Source 1 and Source 2/3):
  NAME×NUM  core-name token + address number  -> survives address reordering, typos in other tokens
  ADDR×NUM  street/locality token + number    -> recovers pseudo-names, native-script and URL names
  NAME×NAME unordered pair of core-name tokens -> recovers empty / number-less addresses
  NAME×ADDR core-name token + street/locality token -> separates same-name branches when numbers are missing
  NOSPACE   whole core name without spaces     -> recovers "orthopedicsafehealth.com"-style names
Optional families (switched on in [blocking]; all off reproduces blocking v2):
  NAME1     one core-name token                -> 2-token names with a typo in the other token, empty address
  ADDR×ADDR unordered pair of street/locality tokens -> pseudo-names whose house number was rewritten
  num_del   Source-1 number with one digit deleted ("2677" -> "677"), the commonest number noise
Keys shared by more than `s1_key_cap` Source-1 records are dropped (non-selective).
Each Source-2/3 record keeps its `top_per_record` Source-1 entities ranked by rarity-weighted key votes
(a key shared by n Source-1 records contributes 1/log2(1+n)):
since every record belongs to at most one entity, ranking from the record side keeps the set small.
"""
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from .utils import log

NAME_W, ADDR_W, PAIR_W = 5, 6, 5      # max tokens used per record for each key family
T_NAME_NUM, T_ADDR_NUM, T_PAIR, T_NOSPACE, T_NAME_ADDR, T_NAME1, T_ADDR_PAIR = 1, 2, 3, 4, 5, 6, 7
NAME_ADDR_W = 3                        # name tokens combined with address tokens
DEL_W = 3                              # address tokens combined with digit-deleted Source-1 numbers


def opt(cfg, name, default=False):
    return getattr(cfg.blocking, name, default)


def digit_deletions(nums: np.ndarray, width: int = 6) -> np.ndarray:
    """(n, width) matrix: the number with its k-th digit (from the left) deleted; -1 when absent."""
    s = nums.astype(np.int64)
    ndig = np.floor(np.log10(np.maximum(s, 1))).astype(np.int64) + 1
    out = np.full((len(s), width), -1, np.int64)
    for k in range(width):
        p = np.clip(ndig - 1 - k, 0, 18)                    # power of ten of the deleted digit
        v = (s // 10 ** (p + 1)) * 10 ** p + s % 10 ** p
        ok = (s >= 10) & (ndig > k)
        out[ok, k] = v[ok]
    return out


def _pad(lst: pa.ListArray, dictionary: pa.Array, width: int) -> np.ndarray:
    """(n, width) int64 matrix of token ids in `dictionary` (-1 = empty / unknown token)."""
    lst = lst.combine_chunks() if isinstance(lst, pa.ChunkedArray) else lst
    ids = pc.fill_null(pc.index_in(lst.flatten(), value_set=dictionary), -1).to_numpy().astype(np.int64)
    off = lst.offsets.to_numpy()
    lens = np.diff(off)
    m = np.full((len(lst), width), -1, np.int64)
    for j in range(width):
        has = lens > j
        m[has, j] = ids[off[:-1][has] + j]
    return m


def _keys(name, addr, nums, nospace, cfg=None, dels=None):
    """Return (row, key) int arrays. Keys pack family<<58 | a<<29 | b (ids and numbers < 2^29)."""
    rows, keys = [], []

    def emit(mask, fam, a, b):
        r = np.flatnonzero(mask)
        rows.append(r)
        keys.append((np.int64(fam) << 58) | (a[r] << 29) | b[r])

    nums = np.where(nums >= 0, np.minimum(nums, (1 << 29) - 1), -1).astype(np.int64)
    for toks, fam in ((name, T_NAME_NUM), (addr, T_ADDR_NUM)):
        for i in range(toks.shape[1]):
            for j in range(nums.shape[1]):
                emit((toks[:, i] >= 0) & (nums[:, j] >= 0), fam, toks[:, i], nums[:, j])
    for i in range(PAIR_W):
        for j in range(i + 1, PAIR_W):
            a, b = np.minimum(name[:, i], name[:, j]), np.maximum(name[:, i], name[:, j])
            emit((a >= 0) & (a != b), T_PAIR, a, b)
    for i in range(NAME_ADDR_W):
        for j in range(addr.shape[1]):
            emit((name[:, i] >= 0) & (addr[:, j] >= 0), T_NAME_ADDR, name[:, i], addr[:, j])
    emit(nospace >= 0, T_NOSPACE, nospace, np.zeros_like(nospace))
    if cfg is not None and opt(cfg, "name1_key"):
        for i in range(name.shape[1]):
            emit(name[:, i] >= 0, T_NAME1, name[:, i], np.zeros_like(nospace))
    if cfg is not None and opt(cfg, "addr_pair_key"):
        for i in range(addr.shape[1]):
            for j in range(i + 1, addr.shape[1]):
                a, b = np.minimum(addr[:, i], addr[:, j]), np.maximum(addr[:, i], addr[:, j])
                emit((a >= 0) & (a != b), T_ADDR_PAIR, a, b)
    if dels is not None:                                     # Source-1 side: number with one digit deleted
        dels = np.where(dels >= 0, np.minimum(dels, (1 << 29) - 1), -1)
        for i in range(DEL_W):
            for j in range(dels.shape[1]):
                emit((addr[:, i] >= 0) & (dels[:, j] >= 0), T_ADDR_NUM, addr[:, i], dels[:, j])
    return np.concatenate(rows), np.concatenate(keys)


class Index:
    """Sorted inverted index over one country's Source-1 records."""

    def __init__(self, s1: pa.Table, rows: np.ndarray, cfg):
        self.cfg, self.rows = cfg, rows
        t = s1.take(rows)
        self.name_dict = pc.unique(t["name_tokens"].combine_chunks().flatten())
        self.addr_dict = pc.unique(t["addr_tokens"].combine_chunks().flatten())
        self.ns_dict = pc.unique(t["name_nospace"].combine_chunks())
        r, k = self._record_keys(t, source1=True)
        order = np.argsort(k, kind="stable")
        k, r = k[order], r[order]
        self.ukeys, self.start, self.count = np.unique(k, return_index=True, return_counts=True)
        self.vals = r.astype(np.int32)
        keep = self.count <= cfg.blocking.s1_key_cap
        log(f"  index: {len(rows):,} S1, {len(k):,} keys, {keep.mean():.1%} of distinct keys selective")
        self.ukeys, self.start, self.count = self.ukeys[keep], self.start[keep], self.count[keep]

    def _record_keys(self, t, source1=False):
        nums = np.stack([t[f"num{j}"].to_numpy() for j in range(self.cfg.blocking.number_keys)], 1)
        ns = pc.fill_null(pc.index_in(t["name_nospace"], value_set=self.ns_dict), -1).to_numpy().astype(np.int64)
        dels = digit_deletions(t["num0"].to_numpy()) if source1 and opt(self.cfg, "num_del_key") else None
        return _keys(_pad(t["name_tokens"], self.name_dict, NAME_W),
                     _pad(t["addr_tokens"], self.addr_dict, ADDR_W), nums, ns, self.cfg, dels)

    def query(self, c: pa.Table):
        """(local cand row, local S1 position, votes) for every cand record's top-k Source-1 entities."""
        r, k = self._record_keys(c)
        pos = np.searchsorted(self.ukeys, k)
        pos[pos == len(self.ukeys)] = 0
        hit = self.ukeys[pos] == k
        r, pos = r[hit], pos[hit]
        cnt = self.count[pos]
        rr = np.repeat(r, cnt)
        base = np.repeat(self.start[pos] - np.cumsum(cnt) + cnt, cnt)       # vectorised range expansion
        s = self.vals[base + np.arange(len(rr))]
        w = np.repeat((1.0 / np.log2(1.0 + cnt)).astype(np.float32), cnt)
        code, inv = np.unique(rr.astype(np.int64) * len(self.rows) + s, return_inverse=True)
        votes = np.bincount(inv, weights=w).astype(np.float32)
        cr, s = code // len(self.rows), code % len(self.rows)
        order = np.lexsort((-votes, cr))                                    # per record, best votes first
        cr, s, votes = cr[order], s[order], votes[order]
        first = np.r_[0, np.flatnonzero(np.diff(cr)) + 1]
        rank = np.arange(len(cr)) - np.repeat(first, np.diff(np.r_[first, len(cr)]))
        best = np.repeat(votes[first], np.diff(np.r_[first, len(cr)]))
        keep = (rank < self.cfg.blocking.top_per_record) & (votes >= self.cfg.blocking.min_rel_votes * best)
        return cr[keep].astype(np.int32), s[keep].astype(np.int32), votes[keep], rank[keep].astype(np.int8)


def generate(s1: pa.Table, cand: pa.Table, cfg):
    """All candidate pairs as dict of arrays: s1 (global row), c (global cand row), votes."""
    out = {"s1": [], "c": [], "votes": [], "rank": []}
    s1_ctry = s1["country"].to_numpy(zero_copy_only=False)
    c_ctry = cand["country"].to_numpy(zero_copy_only=False)
    for country in sorted(set(s1_ctry)):
        rows = np.flatnonzero(s1_ctry == country)
        idx = Index(s1, rows, cfg)
        crow = np.flatnonzero(c_ctry == country)
        for i in range(0, len(crow), cfg.blocking.chunk_size):
            part = crow[i:i + cfg.blocking.chunk_size]
            cr, s, v, rk = idx.query(cand.take(part))
            out["s1"].append(rows[s].astype(np.int32))
            out["c"].append(part[cr].astype(np.int32))
            out["votes"].append(v)
            out["rank"].append(rk)
        log(f"  {country}: {sum(len(x) for x in out['c']):,} pairs so far")
    return {k: np.concatenate(v) for k, v in out.items()}
