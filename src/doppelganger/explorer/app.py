"""Doppelgänger explorer: "why was this matched, and why were the look-alikes rejected?"

Run:  python -m doppelganger explore   (reads artifacts/evidence*.parquet, artifacts/models/stage2.txt, artifacts/validation/)
"""
import json
import math
import re
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse

from .. import config

LABELS = {  # feature -> plain-language evidence name (used for SHAP bars)
    "nm_tset": "name token overlap", "nm_tsort": "full name incl. legal form", "nm_ratio_ns": "name spelling",
    "nm_partial_ns": "name contained in other", "nm_jw": "name prefix similarity", "ad_tset": "address word overlap",
    "ad_tsort": "address order-insensitive", "num_p_in": "house number present", "num_p_first": "house number leads",
    "num_frac1": "share of S1 numbers kept", "num_frac2": "share of numbers that are S1's", "num_logdiff": "house-number shift",
    "num_drop": "house number with dropped digit", "n_nums1": "numbers in S1 address", "n_nums2": "numbers in record",
    "loc_overlap": "same city/region", "c_f_alias": "alias marker (aka/f.k.a./DBA)", "c_f_url": "name written as URL",
    "c_f_junk": "junk prefix", "c_f_script": "native-script name", "c_f_empty_addr": "empty address",
    "c_f_pseudo": "generated pseudo-name", "tok_diff": "extra/missing name words", "n_tok1": "name length",
    "c_s3": "from Source 3", "votes": "blocking key votes", "e_rank": "rank among entity's candidates",
    "e_rel": "score relative to entity's best", "e_n": "entity's candidate count", "e_sum": "entity's expected matches",
    "e_sib": "look-alikes around entity", "c_rank": "rank among record's entities", "c_n": "entities competing for record",
    "c_margin": "margin over rival entity", "e_margin": "margin over next candidate",
}
STATIC = Path(__file__).with_name("static")


class Store:
    def __init__(self, cfg):
        w = cfg.paths.work_dir
        self.ev = pd.read_parquet(w / "evidence.parquet")
        self.fx = pd.read_parquet(w / "evidence_features.parquet")
        self.booster = lgb.Booster(model_file=str(w / "models" / "stage2.txt"))
        unl = w / "models_unlabelled" / "stage2.txt"        # scored countries that had no training labels
        self.booster_unl = lgb.Booster(model_file=str(unl)) if unl.exists() else self.booster
        self.labelled = set(self.ev.loc[self.ev.split == "train", "country"].unique())
        self.groups = self.ev.groupby("s1_id").indices
        ents = self.ev.drop_duplicates("s1_id")
        self.entities = ents[["s1_id", "s1_name", "s1_addr", "country", "split"]].reset_index(drop=True)
        self.demos = self._demos()
        v = w / "validation"
        read = lambda n: pd.read_csv(v / n, index_col=0) if (v / n).exists() else None
        self.comparison, self.slices, self.loco = read("comparison.csv"), read("final_slices.csv"), read("loco_india.csv")
        bench = w / "benchmark.json"
        self.bench = json.loads(bench.read_text()) if bench.exists() else None

    def _demos(self):
        e, f = self.ev, self.fx
        hold = e.split == "train"
        has_true = e[hold & (e.label == 1) & e.selected.fillna(False)].s1_id.unique()
        size = e.groupby("s1_id").size()
        def pick(m):   # entities with 3-8 candidates first: the whole story fits on one screen
            ids = pd.Index(e[m].s1_id.unique()).intersection(has_true)
            n = size.reindex(ids)
            return n[(n >= 3) & (n <= 8)].index.tolist() + n[(n < 3) | (n > 8)].index.tolist()
        lookalike = hold & (e.label == 0) & (f.nm_tset >= 90) & ~e.selected.fillna(False).astype(bool) & (e.p1 >= 0.5)
        return {
            "lookalike": pick(lookalike & (f.num_p_in == 0)),
            "lowsim": pick(hold & (e.label == 1) & e.selected.fillna(False).astype(bool) & (f.nm_tset < 50)),
            "empty": pick(hold & (e.label == 1) & e.selected.fillna(False).astype(bool) & (f.c_f_empty_addr == 1)),
            "branch": pick(hold & (e.label == 0) & (e.rival_id != "") & (f.nm_tset >= 95)),
            "france": e[(e.country == "France") & e.selected.fillna(False).astype(bool)].s1_id.unique().tolist(),
        }


def _num(x):
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else float(x)


def _nums(addr):
    return [str(int(x)) for x in re.findall(r"\d+", addr or "")]


def number_diff(s1_addr, c_addr):
    """Source-1 address numbers missing from the record, and the record's numbers not in Source 1."""
    a, b = _nums(s1_addr), _nums(c_addr)
    return [x for x in dict.fromkeys(a) if x not in b], [x for x in dict.fromkeys(b) if x not in a]


def reasons(r, f, s1_addr="") -> list:
    """Plain-language evidence for one candidate (sign: + supports the match, - argues against, · context)."""
    out, matched = [], bool(r.selected)
    missing, extra = number_diff(s1_addr, r.c_addr) if r.c_addr else ([], [])
    if not matched and missing and extra and f.nm_tset >= 85 and not r.rival_id:
        out.append(["-", f"Source-1 number {', '.join(missing[:2])} absent; record has {', '.join(extra[:3])} instead"
                         " — the look-alike signature"])
    elif f.num_p_in == 1:
        out.append(["+", "Source-1 house number appears in this record"])
        if f.num_frac1 < 1:
            out.append(["-", f"only {f.num_frac1:.0%} of the Source-1 address numbers are present"])
    elif f.num_p_in == 0 and np.isfinite(f.num_logdiff):
        d = int(round(math.expm1(f.num_logdiff)))
        if f.num_drop and matched:
            out.append(["+", "house number with a dropped digit (noise)"])
        elif 0 < d <= 60 and f.nm_tset >= 85 and not matched:
            out.append(["-", f"house number shifted by {d} — the look-alike signature"])
        elif f.num_frac1 >= 0.5:
            out.append(["·", f"primary number differs by {d}, but most other numbers agree"])
        else:
            out.append(["-", f"house number differs by {d}"])
    elif np.isnan(f.num_p_in):
        out.append(["·", "no house number to compare"])
    if f.tok_diff > 0 and f.nm_tset >= 85 and not matched:
        out.append(["-", "extra word in the name (decoys add a descriptor)"])
    flags = [("c_f_pseudo", "·", "name replaced by a generated pseudo-word: identity carried by the address"),
             ("c_f_alias", "+", "alias marker: the real name follows 'aka / f.k.a. / DBA / t/a'"),
             ("c_f_url", "·", "name written as a web address / handle"),
             ("c_f_script", "·", "name in a native script (transliterated with the learned dictionary)"),
             ("c_f_empty_addr", "·", "empty address: name-only evidence")]
    out += [[sg, txt] for col, sg, txt in flags if f[col] == 1]
    if f.nm_tset >= 95:
        out.append(["+", "same business name"])
    elif f.nm_tset < 60 and not (f.c_f_pseudo or f.c_f_url or f.c_f_script):
        out.append(["-", f"names differ (token overlap {f.nm_tset:.0f}/100)"])
    if f.loc_overlap == 0:
        out.append(["-", "different city / region"])
    if r.rival_id:
        out.append(["-", f"record is claimed more strongly by {r.rival_id}"])
    if matched and f.e_sib >= 1 and f.num_p_in == 1:
        out.append(["+", f"wins over {int(f.e_sib)} look-alike(s) of this entity on the house number"])
    return out


def verdict(r):
    if r.selected:
        return "matched"
    if r.rival_id:
        return "claimed by another entity"
    return "rejected"


def create_app(cfg=None) -> FastAPI:
    cfg = cfg or config.load()
    st = Store(cfg)
    app = FastAPI(title="Doppelgänger explorer")

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    @app.get("/api/summary")
    def summary():
        tbl = lambda d: None if d is None else json.loads(d.round(4).to_json(orient="split"))
        return {"comparison": tbl(st.comparison), "slices": tbl(st.slices), "loco": tbl(st.loco),
                "benchmark": st.bench, "demo_counts": {k: len(v) for k, v in st.demos.items()},
                "entities": len(st.entities)}

    @app.get("/api/demo/{kind}")
    def demo(kind: str, seed: int = Query(0, ge=0)):
        ids = st.demos.get(kind)
        if not ids:
            raise HTTPException(404, f"no demo cases for '{kind}'")
        return {"s1_id": ids[seed % len(ids)]}

    @app.get("/api/search")
    def search(q: str = Query(..., min_length=2, max_length=80)):
        pat = re.escape(q.strip())
        e = st.entities
        hit = e[e.s1_name.str.contains(pat, case=False, regex=True) | e.s1_id.str.fullmatch(pat, case=False)]
        return hit.head(20).to_dict(orient="records")

    @app.get("/api/entity/{s1_id}")
    def entity(s1_id: str):
        rows = st.groups.get(s1_id)
        if rows is None:
            raise HTTPException(404, "entity not in the evidence store")
        ev, fx = st.ev.iloc[rows], st.fx.iloc[rows]
        ok = ev.c_id.notna().to_numpy()
        head = ev.iloc[0]
        cands = []
        if ok.any():
            bst = st.booster if head.country in st.labelled else st.booster_unl
            names = bst.feature_name()
            contrib = bst.predict(fx[ok][names], pred_contrib=True)
            for (_, r), (_, f), c in zip(ev[ok].iterrows(), fx[ok].iterrows(), contrib):
                top = np.argsort(-np.abs(c[:-1]))[:6]
                cands.append({
                    "id": r.c_id, "name": r.c_name, "addr": r.c_addr, "p1": _num(r.p1), "p2": _num(r.p2),
                    "verdict": verdict(r), "truth": None if r.label < 0 else int(r.label),
                    "rival": r.rival_id or None, "reasons": reasons(r, f, head.s1_addr),
                    "shap": [[LABELS.get(names[i], names[i]), float(c[i])] for i in top],
                    "nm": _num(f.nm_tset), "ad": _num(f.ad_tset),
                })
            cands.sort(key=lambda x: -x["p2"])
        fooled = [c for c in cands if c["p1"] >= 0.5 and c["verdict"] != "matched"]
        return {"id": head.s1_id, "name": head.s1_name, "addr": head.s1_addr, "country": head.country,
                "split": "validation holdout (ground truth shown)" if head.split == "train" else "test (no labels)",
                "candidates": cands, "fooled": len(fooled),
                "correct": None if head.split != "train" else sum((c["verdict"] == "matched") == (c["truth"] == 1) for c in cands)}

    return app
