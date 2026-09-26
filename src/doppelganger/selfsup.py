"""Self-supervision for unlabelled countries: build the synthetic training block and validate it honestly.

Isolation of test-derived information:
  * uses test *inputs* only (Source-1 France records as generation seeds; France Source-2/3 names/addresses for
    vocabulary; the lexicon's France équivalences from unlabelled pseudo-pairs). No test labels exist or are used.
  * operator RATES come from labelled train pairs of the other countries only.
"""
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc

from . import blocking, features, io, normalize as nz, pipeline, synth
from .utils import log, stage

S1_COLS = list(dict.fromkeys(pipeline.BLOCK_COLS + pipeline.FEAT_COLS))


def _raw(cfg, split, which):
    if which == "s1":
        return io.read_raw(io.raw_path(cfg, split, 1))
    return pa.concat_tables([io.read_raw(io.raw_path(cfg, split, 2)), io.read_raw(io.raw_path(cfg, split, 3))])


def build_profile(cfg, sources: list, split: str, target: str) -> synth.Profile:
    rng = np.random.default_rng(cfg.seed)
    s1, cand = pipeline.load(cfg, "train", ["entity_id", "country"])
    gt = io.ground_truth(cfg, s1["entity_id"], cand["entity_id"])
    ctry = s1["country"].to_numpy(zero_copy_only=False)
    src = np.flatnonzero(np.isin(ctry[gt[0]], sources))
    idx = rng.choice(src, min(150_000, len(src)), replace=False)
    rates = synth.measure_rates(_raw(cfg, "train", "s1").take(gt[0][idx]).to_pandas(),
                                _raw(cfg, "train", "c").take(gt[1][idx]).to_pandas())
    ents = np.flatnonzero(np.isin(ctry, sources))
    n_var = np.bincount(gt[0], minlength=len(ctry))[ents]
    # decoy house-number shifts: non-matching look-alike candidates in the source countries
    pairs = pipeline.candidates(cfg, "train")
    X = pipeline.pair_features(cfg, "train")[["nm_tset", "num_p_in", "num_logdiff"]]
    y, _ = pipeline.labels(cfg, s1["entity_id"], cand["entity_id"], pairs["s1"].to_numpy(), pairs["c"].to_numpy())
    m = (y == 0) & (X.nm_tset.values >= 85) & (X.num_p_in.values == 0) & np.isin(ctry[pairs["s1"].to_numpy()], sources)
    delta = np.rint(np.expm1(X.num_logdiff.values[m & np.isfinite(X.num_logdiff.values)]))
    delta = delta[(delta >= 1) & (delta <= 500)]

    t_s1, t_c = _raw(cfg, split, "s1"), _raw(cfg, split, "c")
    t_s1 = t_s1.filter(pc.equal(t_s1["country"], target)).to_pandas()
    keep = pc.equal(t_c["country"], target)
    t_cf = pipeline.load(cfg, split, ["f_pseudo"])[1]["f_pseudo"].filter(keep).to_numpy(zero_copy_only=False)
    t_c = t_c.filter(keep).to_pandas()
    lex = pipeline.get_lexicon(cfg)
    voc = synth.target_vocabulary(t_s1, t_c, t_cf, lex, target)
    decoys = float(np.clip(len(t_c) / len(t_s1) - n_var.mean(), 0.2, 3.0))
    prof = synth.Profile(rates=rates, n_variants=rng.choice(n_var, 20_000).tolist(),
                         decoy_delta=rng.choice(delta, min(20_000, len(delta))).tolist(),
                         decoys_per_entity=decoys, **voc)
    log(f"synthetic profile {sources}->{target}: decoys/entity={decoys:.2f}, "
        f"rates={ {k: round(v, 3) for k, v in rates.items()} }")
    prof.save(cfg.paths.work_dir / f"synth_profile_{target}.json")
    return prof


def synthetic_block(cfg, prof, split, country, seed_rows=None, tag="SYN"):
    """Generate synthetic records for `country`, block them against the real Source-1 index, featurise, label."""
    lex = pipeline.get_lexicon(cfg)
    s1, _ = pipeline.load(cfg, split, S1_COLS)
    raw1 = _raw(cfg, split, "s1").to_pandas()
    rows = np.flatnonzero(raw1.country.values == country)
    seeds = raw1.iloc[rows if seed_rows is None else seed_rows]
    with stage(f"synthetic block ({country}, {split})"):
        df = synth.Generator(prof, cfg.seed).generate(seeds, cfg.synth.max_entities, tag)
        c = nz.finalize(nz.base_normalize(synth.to_table(df), cfg.prep.max_numbers), lex)
        cr, s, v, _ = blocking.Index(s1, rows, cfg).query(c)
        ps = rows[s]
        X = features.pair_features(s1, c, ps, cr, v, cfg)
        origin, decoy = df.origin.to_numpy(), df.decoy.to_numpy()
        y = ((ps == origin[cr]) & ~decoy[cr]).astype(np.int8)
        pos = ~decoy
        found = np.zeros(len(df), bool)
        found[cr[y == 1]] = True
        log(f"  {len(df):,} synthetic records ({decoy.mean():.0%} decoys) -> {len(y):,} pairs, "
            f"{y.mean():.1%} positive, synthetic blocking recall {found[pos].mean():.3f}")
    return {"X": X, "y": y, "s1": ps.astype(np.int64), "c": cr.astype(np.int64),
            "table": c.select(features.COHERENCE_COLS)}


PSEUDO_S1_OFFSET = 1 << 30   # keeps pseudo-labelled groups apart from synthetic groups seeded by the same entity


def pseudo_block(cfg, split, country, p2):
    """Self-training block: real candidate pairs of `country`, labelled by the current model's own decision.

    Only entities whose every candidate is confidently decided (p >= hi or p <= lo) are used, so competition
    groups stay complete and pseudo-label noise stays small. No true labels are read."""
    sp = cfg.synth
    s1, _ = pipeline.load(cfg, split, ["country"])
    pairs = pipeline.candidates(cfg, split)
    ps, pcand = pairs["s1"].to_numpy(), pairs["c"].to_numpy()
    sel = pipeline.final_decision(cfg, ps, pcand, p2)
    unsure = np.bincount(ps, weights=((p2 < sp.pseudo_hi) & (p2 > sp.pseudo_lo)), minlength=s1.num_rows)
    has = np.bincount(ps, minlength=s1.num_rows) > 0
    ctry = s1["country"].to_numpy(zero_copy_only=False)
    ents = np.flatnonzero((ctry == country) & has & (unsure == 0))
    rng = np.random.default_rng(cfg.seed)
    ents = rng.choice(ents, min(sp.pseudo_entities, len(ents)), replace=False)
    m = np.isin(ps, ents)
    X = pipeline.model_view(cfg, pipeline.pair_features(cfg, split))[m].reset_index(drop=True)
    y = (sel[m] & (p2[m] >= sp.pseudo_hi)).astype(np.int8)
    log(f"  pseudo-labelled block ({country}, {split}): {len(ents):,} confident entities -> {m.sum():,} pairs, "
        f"{y.mean():.1%} positive")
    return {"X": X, "y": y, "s1": ps[m].astype(np.int64) + PSEUDO_S1_OFFSET, "c": pcand[m].astype(np.int64),
            "table": pipeline.load(cfg, split, features.COHERENCE_COLS)[1]}


def merge_blocks(blocks):
    """Concatenate training blocks; candidate ids are offset so each block indexes its own coherence table."""
    blocks = [b for b in blocks if b is not None]
    if not blocks:
        return None
    off = 0
    for b in blocks:
        b["c"] = b["c"] + off
        off += b["table"].num_rows
    out = {k: np.concatenate([b[k] for b in blocks]) for k in ("y", "s1", "c")}
    out["X"] = pd.concat([b["X"] for b in blocks], ignore_index=True)
    out["table"] = pa.concat_tables([b["table"].select(features.COHERENCE_COLS) for b in blocks])
    return out


def loco_experiment(cfg):
    """Pretend India is unlabelled: US-only model vs US + synthetic India (US rates, India vocabulary)."""
    res = {}
    done = lambda tag: (cfg.paths.work_dir / f"scores_train{tag}.parquet").exists()
    if not done("_locoA"):
        pipeline.train(cfg, countries=["US"], tag="_locoA")
    res["A: US labels only"] = pipeline.validate(cfg, "_locoA", ["India"], final_only=True)[1]
    prof = build_profile(cfg, ["US"], "train", "India")
    s1, _ = pipeline.load(cfg, "train", ["entity_id", "country"])
    role = pipeline.split_roles(cfg, s1["entity_id"].to_numpy(zero_copy_only=False))
    seeds = np.flatnonzero((role == 1) & (s1["country"].to_numpy(zero_copy_only=False) == "India"))
    extra = synthetic_block(cfg, prof, "train", "India", seeds)
    if not done("_locoB"):
        pipeline.train(cfg, extra=extra, countries=["US"], tag="_locoB")
    res["B: US labels + synthetic India"] = pipeline.validate(cfg, "_locoB", ["India"], final_only=True)[1]
    if not done("_locoS"):
        p2 = pipeline.final_prob(pd.read_parquet(cfg.paths.work_dir / "scores_train_locoB.parquet"))
        both = merge_blocks([synthetic_block(cfg, prof, "train", "India", seeds), pseudo_block(cfg, "train", "India", p2)])
        pipeline.train(cfg, extra=both, countries=["US"], tag="_locoS")
    res["S: + self-training on India (no labels)"] = pipeline.validate(cfg, "_locoS", ["India"], final_only=True)[1]
    res["C: real India labels (upper bound)"] = pipeline.validate(cfg, "", ["India"], final_only=True)[1]
    table = pd.DataFrame({k: v.loc["ALL"] for k, v in res.items()}).T
    table.to_csv(cfg.paths.work_dir / "validation" / "loco_india.csv")
    log("LOCO (India holdout, India treated as unlabelled):\n" + table.round(4).to_string())
    return table


def france_block(cfg, p2=None):
    """Training block for every test country without training labels (France in this challenge): synthetic
    records, plus (given test pair scores p2 from a first-round model) a self-training block."""
    s1_tr, _ = pipeline.load(cfg, "train", ["country"])
    s1_te, _ = pipeline.load(cfg, "test", ["country"])
    labelled = sorted(set(s1_tr["country"].to_numpy(zero_copy_only=False)))
    targets = sorted(set(s1_te["country"].to_numpy(zero_copy_only=False)) - set(labelled))
    blocks = [synthetic_block(cfg, build_profile(cfg, labelled, "test", t), "test", t, tag=f"SYN{t[:2].upper()}")
              for t in targets]
    if p2 is not None:
        blocks += [pseudo_block(cfg, "test", t, p2) for t in targets]
    return merge_blocks(blocks)
