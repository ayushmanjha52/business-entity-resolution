"""Evidence store for the explorer: every candidate of a set of entities with the raw records, the stage-2
feature vector (for on-demand SHAP), both scores, the rival that claims the record, and the decision."""
import numpy as np
import pandas as pd
import pyarrow as pa

from . import decide, io, pipeline
from .utils import log, stage


def _raw_strings(cfg, split):
    s1 = io.read_raw(io.raw_path(cfg, split, 1))
    c = pa.concat_tables([io.read_raw(io.raw_path(cfg, split, k)) for k in (2, 3)])
    return s1, c


def _build(cfg, split, entities, ps, pcand, X2, p1, p2, sel, y=None):
    own = decide.owner(ps, pcand, p2)
    owner_of = pd.Series(ps[own], index=pcand[own])
    m = np.isin(ps, entities)
    s1raw, craw = _raw_strings(cfg, split)
    e = pd.DataFrame({"s1": ps[m], "c": pcand[m], "p1": p1[m], "p2": p2[m], "selected": sel[m], "owner": own[m]})
    e["rival_s1"] = owner_of.reindex(e.c).to_numpy()
    e.loc[e.rival_s1 == e.s1, "rival_s1"] = -1
    e["label"] = y[m] if y is not None else -1
    feats = X2.loc[m].reset_index(drop=True)
    for col, t, rows in (("s1", s1raw, e.s1), ("c", craw, e.c)):
        sub = t.take(pa.array(rows.to_numpy())).to_pandas()
        e[f"{col}_id"], e[f"{col}_name"], e[f"{col}_addr"] = sub.entity_id.values, sub.business_name.values, sub.business_address.values
    e["country"] = s1raw["country"].take(pa.array(e.s1.to_numpy())).to_numpy(zero_copy_only=False)
    rv = e.rival_s1.to_numpy()
    rid = np.full(len(e), "", object)
    rid[rv >= 0] = s1raw["entity_id"].take(pa.array(rv[rv >= 0])).to_numpy(zero_copy_only=False)
    e["rival_id"] = rid
    e["split"] = split
    # entities with no candidates still need a row so the explorer can show "no candidates"
    lonely = np.setdiff1d(entities, e.s1.unique())
    if len(lonely):
        sub = s1raw.take(pa.array(lonely)).to_pandas()
        e = pd.concat([e, pd.DataFrame({"s1": lonely, "s1_id": sub.entity_id, "s1_name": sub.business_name,
                                        "s1_addr": sub.business_address, "country": sub.country, "split": split,
                                        "label": -1})], ignore_index=True)
        feats = pd.concat([feats, pd.DataFrame(np.nan, index=range(len(lonely)), columns=feats.columns)], ignore_index=True)
    return e, feats


def build(cfg, test_sample=60_000):
    """Write artifacts/evidence.parquet (+ features) for holdout entities and a test sample."""
    parts, fparts = [], []
    with stage("evidence store"):
        s1, cand = pipeline.load(cfg, "train", ["entity_id"])
        pairs = pipeline.candidates(cfg, "train")         # holdout: exactly the scores validation measured
        ps, pcand = pairs["s1"].to_numpy(), pairs["c"].to_numpy()
        sc = pd.read_parquet(cfg.paths.work_dir / "scores_train.parquet")
        p1, p2, y = sc.p1.to_numpy(), sc.p2.to_numpy(), sc.y.to_numpy()
        X2 = pipeline.stage2_matrix(pipeline.pair_features(cfg, "train"), ps, pcand, p1,
                                    pipeline.coherence_table(cfg, "train"))
        role = pipeline.split_roles(cfg, s1["entity_id"].to_numpy(zero_copy_only=False))
        sel = pipeline.final_decision(cfg, ps, pcand, p2)
        e, f = _build(cfg, "train", np.flatnonzero(role == 2), ps, pcand, X2, p1, p2, sel, y)
        parts.append(e), fparts.append(f)
        del X2
        s1t, _ = pipeline.load(cfg, "test", ["entity_id"])
        ps, pcand, X2, p1, p2 = pipeline.score(cfg, "test")
        sel = pipeline.final_decision(cfg, ps, pcand, p2)
        ents = np.random.default_rng(cfg.seed).choice(s1t.num_rows, test_sample, replace=False)
        e, f = _build(cfg, "test", ents, ps, pcand, X2, p1, p2, sel)
        parts.append(e), fparts.append(f)
        ev, fv = pd.concat(parts, ignore_index=True), pd.concat(fparts, ignore_index=True)
        ev.to_parquet(cfg.paths.work_dir / "evidence.parquet")
        fv.astype(np.float32).to_parquet(cfg.paths.work_dir / "evidence_features.parquet")
    log(f"evidence: {ev.s1_id.nunique():,} entities, {len(ev):,} candidate rows")
