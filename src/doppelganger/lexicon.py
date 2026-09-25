"""Equivalences learned ONLY from the provided files (no external gazetteer / transliteration service).

* components  -- address-component variants -> Source-1 canonical form, per country
                 ("tn"->"tamil nadu", "தமிழ்நாடு"->"tamil nadu", "texas"->"tx", "gironde"->"nouvelle aquitaine").
                 Learned from labelled train pairs; for countries without labels (France) from unlabelled
                 test pseudo-pairs (identical core name + shared address number). No test labels exist or are used.
* translit    -- native-script name token -> Latin token, from position-aligned labelled pairs.
* localities  -- frequent canonical components (states, régions, big cities) used as a locality signal.
* vocab       -- Source-1 name vocabulary; a single out-of-vocabulary token marks a generated pseudo-name.
"""
import json
from dataclasses import asdict, dataclass
from functools import cached_property

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc

from . import normalize as nz
from .utils import log


@dataclass
class Lexicon:
    translit: dict
    components: dict
    localities: list
    vocab: list
    pseudo_min_len: int = 5

    @cached_property
    def _vocab(self):
        return pa.array(self.vocab, pa.string())

    @cached_property
    def _loc(self):
        return pa.array(self.localities, pa.string())

    def vocab_array(self):
        return self._vocab

    @cached_property
    def _vocab_set(self):
        return {w for w in self.vocab if len(w) >= 2} | nz.LEGAL

    def vocab_set(self) -> set:
        """Words URL names are segmented into (Source-1 vocabulary + legal forms, dropped later from the core)."""
        return self._vocab_set

    def locality_array(self):
        return self._loc

    def save(self, path):
        path.write_text(json.dumps(asdict(self), ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path):
        return cls(**json.loads(path.read_text(encoding="utf-8")))


def keyed_components(t: pa.Table, rows=None) -> pd.DataFrame:
    """Long table (row, 'country|component') for the given rows (row = position in `rows`)."""
    sub = t.select(["country", "addr_comps"]) if rows is None else t.select(["country", "addr_comps"]).take(rows)
    lst = sub["addr_comps"].combine_chunks()
    par = pc.list_parent_indices(lst)
    key = pc.binary_join_element_wise(pc.take(sub["country"].combine_chunks(), par), lst.flatten(), "|")
    return pd.DataFrame({"row": par.to_numpy(), "comp": key.to_numpy(zero_copy_only=False)}).drop_duplicates()


def learn_components(s1, cand, a, b, s1_rate, c_rate, cfg) -> dict:
    """Map a candidate-side component to the Source-1 component it co-occurs with most specifically."""
    d1, d2 = keyed_components(s1, a), keyed_components(cand, b)
    m = d1.merge(d2, on="row", suffixes=("_1", "_2"))
    df = m.groupby(["comp_2", "comp_1"], sort=False).size().rename("n").reset_index()
    df = df[(df.comp_1 != df.comp_2) & ~df.comp_2.str.contains(r"\d")]   # street lines are not equivalences
    f1, f2 = d1.comp.value_counts(), d2.comp.value_counts()
    df["f2"] = f2.reindex(df.comp_2).values
    df["share"] = df.n / df.f2
    df["cos"] = df.n ** 2 / (df.f2 * f1.reindex(df.comp_1).values)   # prefers the specific equivalent
    best = df.sort_values("cos").drop_duplicates("comp_2", keep="last")
    ratio = s1_rate.reindex(best.comp_2).fillna(0).values / c_rate.reindex(best.comp_2).fillna(1e-9).values
    ok = (best.share >= cfg.component_min_share) & (best.f2 >= cfg.component_min_count) & (ratio <= cfg.foreign_max_ratio)
    best = best[ok]
    return dict(zip(best.comp_2, best.comp_1.str.split("|", n=1).str[1]))


def learn_translit(s1, cand, a, b, cfg) -> dict:
    """Position-align native-script tokens with Latin tokens in labelled pairs of equal token count."""
    sc = cand["f_script"].to_numpy(zero_copy_only=False)[b]
    a, b = a[sc], b[sc]
    t1, t2 = s1["name_tokens"].take(a).combine_chunks(), cand["name_tokens"].take(b).combine_chunks()
    same = pc.equal(pc.list_value_length(t1), pc.list_value_length(t2))
    x1, x2 = t1.filter(same).flatten(), t2.filter(same).flatten()
    df = pd.DataFrame({"lat": x1.to_numpy(zero_copy_only=False), "nat": x2.to_numpy(zero_copy_only=False)})
    df = df[df.nat.str.contains(r"[^\x00-ɏ]") & df.lat.str.fullmatch(r"[a-z0-9]+")]
    c = df.groupby(["nat", "lat"], sort=False).size().rename("n").reset_index()
    tot = c.groupby("nat").n.transform("sum")
    c = c.assign(share=c.n / tot).sort_values("n").drop_duplicates("nat", keep="last")
    c = c[(c.n >= cfg.translit_min_count) & (c.share >= cfg.translit_min_share)]
    return dict(zip(c.nat, c.lat))


def pseudo_pairs(s1, cand, countries, limit, seed):
    """Unlabelled high-precision pairs: identical core name and S1's first number among cand's first 3."""
    def keys(t, j):
        ctry = t["country"].combine_chunks()
        ok = pc.and_(pc.is_in(ctry, value_set=pa.array(countries)), pc.greater_equal(t[f"num{j}"].combine_chunks(), 0))
        t = t.filter(ok)
        _, core = nz.core_name(t["name_tokens"].combine_chunks())
        k = pc.binary_join_element_wise(t["country"].combine_chunks(), core, pc.cast(t[f"num{j}"], pa.string()), "|")
        return pd.DataFrame({"k": k.to_numpy(zero_copy_only=False)}, index=np.flatnonzero(ok.to_numpy(zero_copy_only=False)))

    k1 = keys(s1, 0)
    k1 = k1[~k1.k.duplicated(keep=False)].reset_index(names="s")
    p = pd.concat([keys(cand, j).reset_index(names="c").merge(k1, on="k")[["s", "c"]] for j in range(3)]).drop_duplicates()
    p = p.sample(min(limit, len(p)), random_state=seed)
    return p.s.to_numpy(np.int32), p.c.to_numpy(np.int32)


def fit(cfg, train, test, gt) -> Lexicon:
    """train/test = (s1, cand) base tables; gt = (s1_row, cand_row) labelled pairs of model-training entities only."""
    rng = np.random.default_rng(cfg.seed)
    lc = cfg.lexicon
    idx = rng.choice(len(gt[0]), min(lc.pair_sample, len(gt[0])), replace=False)
    a, b = gt[0][idx], gt[1][idx]
    translit = learn_translit(train[0], train[1], a, b, lc)
    log(f"lexicon: {len(translit):,} transliteration tokens")

    def rates(t):  # component frequency per record, from a sample of both splits
        rows = rng.choice(t.num_rows, min(2_000_000, t.num_rows), replace=False)
        return keyed_components(t, rows).comp.value_counts() / len(rows)
    s1_rate = pd.concat([rates(train[0]), rates(test[0])]).groupby(level=0).max()
    c_rate = pd.concat([rates(train[1]), rates(test[1])]).groupby(level=0).max()
    comps = learn_components(train[0], train[1], a, b, s1_rate, c_rate, lc)
    unlabelled = sorted(set(pc.unique(test[0]["country"]).to_pylist()) - set(pc.unique(train[0]["country"]).to_pylist()))
    if unlabelled:
        pa_, pb_ = pseudo_pairs(test[0], test[1], unlabelled, lc.pseudo_pair_max, cfg.seed)
        log(f"lexicon: {len(pa_):,} unlabelled pseudo-pairs for {unlabelled}")
        comps.update(learn_components(test[0], test[1], pa_, pb_, s1_rate, c_rate, lc))
    log(f"lexicon: {len(comps):,} component equivalences")

    loc = []
    for t in (train[0], test[0]):
        k = keyed_components(t)
        ctry = k.comp.str.split("|", n=1).str[0]
        k["comp"] = (ctry + "|" + k.comp.map(comps)).fillna(k.comp)            # canonicalise first
        n_rec = pd.Series(t["country"].to_numpy(zero_copy_only=False)).value_counts()
        cnt = k.drop_duplicates().groupby("comp").size()
        share = cnt / cnt.index.str.split("|", n=1).str[0].map(n_rec).values
        loc += share[share >= lc.locality_min_share].index.tolist()
    vocab = pc.unique(pa.chunked_array([train[0]["name_tokens"].combine_chunks().flatten(),
                                        test[0]["name_tokens"].combine_chunks().flatten()])).to_pylist()
    log(f"lexicon: {len(set(loc)):,} localities, {len(vocab):,} S1 name tokens")
    return Lexicon(translit, comps, sorted(set(loc)), vocab, lc.pseudo_min_len)
