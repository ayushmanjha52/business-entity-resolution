"""Test-like rivalry for training and evaluation.

The test split carries ~1.1 more Source-2/3 records per Source-1 entity than train (5.75 vs 4.68), while the
number of true matches per entity is unchanged (~3.4): the surplus is sibling decoys (same name and street,
shifted house number). The `train_aug` split is the train split plus synthetic sibling decoys drawn from the
measured noise model, so the models and the VALID/TEST evaluation see test-like candidate crowding.
Synthetic decoys are never labelled true, and no test labels exist or are used.
"""
import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from . import normalize as nz, pipeline, selfsup, synth
from .utils import log, stage


def table_path(cfg):
    return cfg.paths.work_dir / "final" / "train_aug_decoys.parquet"


def extra_decoys_per_entity(cfg) -> float:
    """Test minus train Source-2/3 records per Source-1 entity (true matches per entity are the same)."""
    density = {}
    for split in ("train", "test"):
        s1, cand = pipeline.load(cfg, split, ["country"])
        density[split] = cand.num_rows / s1.num_rows
    return max(density["test"] - density["train"], 0.0)


def build(cfg) -> None:
    extra = extra_decoys_per_entity(cfg)
    raw = selfsup._raw(cfg, "train", "s1").to_pandas()
    lex = pipeline.get_lexicon(cfg)
    parts = []
    with stage(f"test-like sibling decoys (+{extra:.2f} per train entity)"):
        for country in sorted(raw.country.unique()):
            prof = selfsup.build_profile(cfg, [country], "train", country)
            prof.n_variants, prof.decoys_per_entity = [0], extra
            seeds = raw[raw.country == country]
            df = synth.Generator(prof, cfg.seed).generate(seeds, len(seeds), f"AUG{country[:2].upper()}")
            parts.append(nz.finalize(nz.base_normalize(synth.to_table(df), cfg.prep.max_numbers), lex))
            log(f"  {country}: {len(df):,} synthetic decoys for {len(seeds):,} entities")
    pq.write_table(pa.concat_tables(parts), table_path(cfg))


def table(cfg, columns=None) -> pa.Table:
    if not table_path(cfg).exists():
        build(cfg)
    return pq.read_table(table_path(cfg), columns=columns).combine_chunks()


def hard_siblings(X) -> np.ndarray:
    """Same name, same street, house number not kept: the decoys the test split has more of."""
    return ((X.nm_tset >= 85) & (X.ad_tset >= 90) & (X.num_p_in == 0)).to_numpy()


def _sibling_density(cfg, split):
    s1, _ = pipeline.load(cfg, split, ["country"])
    ctry = s1["country"].to_numpy(zero_copy_only=False)
    ps = pipeline.candidates(cfg, split)["s1"].to_numpy()
    hard = hard_siblings(pipeline.pair_features(cfg, split)[["nm_tset", "ad_tset", "num_p_in"]])
    return {c: hard[ctry[ps] == c].sum() / (ctry == c).sum() for c in np.unique(ctry)}


def calibration_path(cfg):
    return cfg.paths.work_dir / "aug_calibration.json"


def calibrate(cfg, pairs: pa.Table, X, n_pairs: int, n_cand: int) -> None:
    """Thin the synthetic decoys per country so train_aug has the test split's hard-sibling density.

    Label-free: compares hard same-street siblings per Source-1 entity in train vs test candidates, and keeps
    each synthetic decoy with the probability that closes that gap. Rewrites the decoy table and the cached
    train_aug pairs / features consistently (blocking and features are per record, so filtering is exact)."""
    train, test = _sibling_density(cfg, "train"), _sibling_density(cfg, "test")
    s1, _ = pipeline.load(cfg, "train", ["country"])
    s1_ctry = s1["country"].to_numpy(zero_copy_only=False)
    aug = table(cfg)
    rec_ctry = aug["country"].to_numpy(zero_copy_only=False)
    ps, pc = pairs["s1"].to_numpy()[n_pairs:], pairs["c"].to_numpy()[n_pairs:] - n_cand
    hard = hard_siblings(X.iloc[n_pairs:])
    rng = np.random.default_rng(cfg.seed)
    keep_rec = np.zeros(aug.num_rows, bool)
    report = {}
    for c in np.unique(rec_ctry):
        have = hard[s1_ctry[ps] == c].sum() / (s1_ctry == c).sum()
        p = float(np.clip((test.get(c, 0) - train.get(c, 0)) / max(have, 1e-9), 0, 1))
        m = rec_ctry == c
        keep_rec[m] = rng.random(m.sum()) < p
        report[c] = {"train": train[c], "test": test.get(c), "synthetic_full": have, "keep": p}
        log(f"  {c}: hard siblings/entity train {train[c]:.3f}, test {test.get(c, 0):.3f}, "
            f"synthetic {have:.3f} -> keep {p:.1%} of decoys")
    new_id = np.cumsum(keep_rec) - 1
    keep_pair = np.r_[np.ones(n_pairs, bool), keep_rec[pc]]
    c_all = pairs["c"].to_numpy().copy()
    c_all[n_pairs:] = n_cand + new_id[pc]
    t = pairs.set_column(pairs.schema.get_field_index("c"), "c", pa.array(c_all, pa.int32())).filter(pa.array(keep_pair))
    pq.write_table(t, cfg.paths.work_dir / f"pairs_{pipeline.AUG}.parquet")
    X[keep_pair].reset_index(drop=True).to_parquet(cfg.paths.work_dir / f"feats_{pipeline.AUG}.parquet")
    pq.write_table(aug.filter(pa.array(keep_rec)), table_path(cfg))
    calibration_path(cfg).write_text(json.dumps(report, indent=2, default=float))
    log(f"calibrated train_aug: {keep_rec.sum():,} of {aug.num_rows:,} synthetic decoys kept, {t.num_rows:,} pairs")
