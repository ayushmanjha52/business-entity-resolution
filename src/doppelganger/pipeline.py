"""Stage orchestration. Every stage caches its artefact in artifacts/ so later stages can be re-run cheaply."""
import json

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from . import blocking, decide, features, io, lexicon, metrics, model
from .utils import log, stage


def lex_path(cfg):
    return cfg.paths.work_dir / "lexicon.json"


def get_lexicon(cfg):
    p = lex_path(cfg)
    if not p.exists():
        with stage("[1/6] learning lexicon (train labels + unlabelled test structure)"):
            train, test = io.load_split(cfg, "train"), io.load_split(cfg, "test")
            gt = io.ground_truth(cfg, train[0]["entity_id"], train[1]["entity_id"])
            fit_rows = split_roles(cfg, train[0]["entity_id"].to_numpy(zero_copy_only=False))[gt[0]] == TRAIN
            lexicon.fit(cfg, train, test, (gt[0][fit_rows], gt[1][fit_rows])).save(p)
            del train, test
    return lexicon.Lexicon.load(p)


AUG = "train_aug"   # train split + synthetic sibling decoys at test-like density (see augment.py)


def fit_split(cfg) -> str:
    """Split that trains and evaluates the models: train, or train_aug when split.decoy_augment is on."""
    return AUG if getattr(cfg.split, "decoy_augment", False) else "train"


def load(cfg, split, columns=None):
    if split == AUG:
        from . import augment
        _ensure_aug(cfg)
        s1, cand = io.load_split(cfg, "train", get_lexicon(cfg), columns)
        return s1, pa.concat_tables([cand, augment.table(cfg, columns or cand.column_names)]).combine_chunks()
    return io.load_split(cfg, split, get_lexicon(cfg), columns)


def _aug_part(cfg, columns):
    """(train Source-1 table, synthetic decoy table, number of real train candidates)."""
    from . import augment
    s1, cand = io.load_split(cfg, "train", get_lexicon(cfg), columns)
    return s1, augment.table(cfg, columns), cand.num_rows


def _ensure_aug(cfg):
    """Build train_aug once: block and featurise the synthetic decoys (per record, so independent of the real
    records), then thin them to the test split's hard-sibling density. Only calibrated files are ever read."""
    from . import augment
    if augment.calibration_path(cfg).exists():
        return
    pp, fp = (cfg.paths.work_dir / f"{k}_{AUG}.parquet" for k in ("pairs", "feats"))
    real = candidates(cfg, "train")
    s1, aug, n_real = _aug_part(cfg, list(dict.fromkeys(BLOCK_COLS + FEAT_COLS)))
    if pp.exists():
        pairs = pq.read_table(pp)
    else:
        with stage("[2/6] candidate generation (synthetic decoys)"):
            extra = blocking.generate(s1, aug, cfg)
        extra["c"] = extra["c"] + n_real
        pairs = pa.concat_tables([real, pa.table(extra).cast(real.schema)])
        pq.write_table(pairs, pp)
    if fp.exists():
        X = pd.read_parquet(fp)
    else:
        part = pairs.slice(real.num_rows)
        with stage(f"[3/6] pair features (synthetic decoys, {part.num_rows:,} pairs)"):
            Xa = features.pair_features(s1, aug, part["s1"].to_numpy(), part["c"].to_numpy() - n_real,
                                        part["votes"].to_numpy(), cfg)
        X = pd.concat([pair_features(cfg, "train"), Xa], ignore_index=True)
        X.to_parquet(fp)
    augment.calibrate(cfg, pairs, X, real.num_rows, n_real)


UNUSED, TRAIN, VALID, TEST = 0, 1, 2, 3


def split_roles(cfg, s1_ids) -> np.ndarray:
    """Deterministic per-entity role from a hash of the id: VALID = [0, valid_frac) (model/threshold selection),
    TEST = [1 - test_frac, 1) (scored once, at the end), TRAIN = the next train_frac after VALID."""
    u = (pd.util.hash_array(np.asarray(s1_ids, dtype=object), hash_key="doppelganger0000") % 10_000) / 10_000
    sp = cfg.split
    assert sp.valid_frac + sp.train_frac <= 1 - sp.test_frac, "split fractions overlap"
    role = np.full(len(u), UNUSED, np.int8)
    role[u < sp.valid_frac] = VALID
    role[(u >= sp.valid_frac) & (u < sp.valid_frac + sp.train_frac)] = TRAIN
    role[u >= 1 - sp.test_frac] = TEST
    return role


BLOCK_COLS = ["entity_id", "country", "name_tokens", "name_nospace", "addr_tokens", "num0", "num1", "num2"]


def candidates(cfg, split):
    p = cfg.paths.work_dir / f"pairs_{split}.parquet"
    if p.exists():
        return pq.read_table(p)
    if split == AUG:
        _ensure_aug(cfg)
        return pq.read_table(p)
    s1, cand = load(cfg, split, BLOCK_COLS)
    with stage(f"[2/6] candidate generation ({split})"):
        pairs = blocking.generate(s1, cand, cfg)
    t = pa.table(pairs)
    pq.write_table(t, p)
    log(f"{split}: {t.num_rows:,} candidate pairs ({t.num_rows / s1.num_rows:.1f} per S1, "
        f"{t.num_rows / cand.num_rows:.2f} per S2/S3 record; reduction ratio "
        f"{1 - t.num_rows / (s1.num_rows * cand.num_rows):.8f})")
    return t


def blocking_report(cfg):
    """Candidate recall and oracle F0.5 ceiling on the full train split."""
    s1, cand = load(cfg, "train", ["entity_id", "country"])
    gt = io.ground_truth(cfg, s1["entity_id"], cand["entity_id"])
    pairs = candidates(cfg, "train")
    ps, pc_ = pairs["s1"].to_numpy(), pairs["c"].to_numpy()
    rank, votes = pairs["rank"].to_numpy(), pairs["votes"].to_numpy()
    rel = votes / pd.Series(votes).groupby(pc_).transform("max").to_numpy()
    is_true = np.isin(metrics.pair_code(ps, pc_, cand.num_rows), metrics.pair_code(*gt, cand.num_rows))
    curve = [(f"rank<{k}", (rank < k)) for k in (1, 2, 3, 4, 6, 8)] + [(f"rel>={r}", rel >= r) for r in (0.2, 0.35, 0.5)]
    log("recall/cost curve:\n" + "\n".join(f"  {n:9s} pairs={m.sum():>11,}  recall={is_true[m].sum() / len(gt[0]):.4f}"
                                           for n, m in curve))
    hit = np.isin(metrics.pair_code(*gt, cand.num_rows), metrics.pair_code(ps, pc_, cand.num_rows))
    oracle = metrics.per_entity((gt[0][hit], gt[1][hit]), gt, np.arange(s1.num_rows), cand.num_rows)
    ctry = pd.Series(s1["country"].to_numpy(zero_copy_only=False))
    rep = metrics.report(oracle, {"country": ctry})
    log(f"pair recall {hit.mean():.4f}; oracle ceiling:\n{rep[['entities', 'F0.5', 'recall']].to_string()}")
    return hit


# ----------------------------------------------------------------------------------------------- features
FEAT_COLS = list(dict.fromkeys(features.PAIR_COLS + [f"num{j}" for j in range(6)]))
SIB_NAME = 90   # "look-alike": name similarity at least this, but the S1 house number is absent


def pair_features(cfg, split) -> pd.DataFrame:
    p = cfg.paths.work_dir / f"feats_{split}.parquet"
    if split == AUG:
        _ensure_aug(cfg)
    if not p.exists():
        s1, cand = load(cfg, split, FEAT_COLS)
        pairs = candidates(cfg, split)
        with stage(f"[3/6] pair features ({split}, {pairs.num_rows:,} pairs)"):
            X = features.pair_features(s1, cand, pairs["s1"].to_numpy(), pairs["c"].to_numpy(),
                                       pairs["votes"].to_numpy(), cfg)
        X.to_parquet(p)
        return X
    return pd.read_parquet(p)


def model_view(cfg, X: pd.DataFrame) -> pd.DataFrame:
    """Columns the models see (feature-store columns listed in model.exclude stay available for explanations)."""
    return X.drop(columns=[c for c in getattr(cfg.model, "exclude", []) if c in X.columns])


def sibling_flag(X) -> np.ndarray:
    return ((X.nm_tset >= SIB_NAME) & (X.num_p_in == 0)).to_numpy(np.float32)


def rival_matrix(X, s1, c, p, cand=None, prefix="") -> pd.DataFrame:
    """X + competition features (+ group-coherence features when a cand table is given) computed from scores p."""
    parts = [features.competition_features(s1, c, p, sibling_flag(X))]
    if cand is not None:
        parts.append(features.coherence_features(s1, c, p, cand))
    return pd.concat([X.reset_index(drop=True)] + [f.add_prefix(prefix) for f in parts], axis=1)


def stage2_matrix(X, s1, c, p1, cand=None) -> pd.DataFrame:
    return rival_matrix(X, s1, c, p1, cand)


def stage3_matrix(X2, s1, c, p2, cand=None) -> pd.DataFrame:
    """Stage 3 re-reads the rivalry with the sharper stage-2 scores (VALID: 0.9799 -> 0.9808)."""
    return rival_matrix(X2, s1, c, p2, cand, prefix="s3_")


def final_prob(sc: pd.DataFrame) -> np.ndarray:
    """Probability of the last stage stored in a scores file."""
    return (sc["p"] if "p" in sc else sc["p2"]).to_numpy()


def coherence_table(cfg, split):
    """Candidate-record columns for group-coherence features (None when disabled in config)."""
    return load(cfg, split, features.COHERENCE_COLS)[1] if cfg.model.coherence else None


def labels(cfg, s1_ids, cand_ids, ps, pcand) -> np.ndarray:
    gt = io.ground_truth(cfg, s1_ids, cand_ids)
    return np.isin(metrics.pair_code(ps, pcand, len(cand_ids)), metrics.pair_code(*gt, len(cand_ids))).astype(np.int8), gt


# ----------------------------------------------------------------------------------------------- training
def train(cfg, extra=None, countries=None, tag="", roles=(TRAIN,)):
    """Stage 1 (pairwise), stage 2 (+ competition and coherence from out-of-fold stage-1 scores) and, when
    model.stages == 3, stage 3 (the same rival features recomputed from out-of-fold stage-2 scores).

    roles:     entity roles whose labelled pairs train the models (development: TRAIN; final model: all labelled).
    countries: restrict real training pairs to these countries (leave-one-country-out experiments).
    extra:     optional synthetic block {X, y, s1, c} (generator inversion) appended to the training data;
               its stage-1 scores are out-of-fold too, so stage-2 sees them exactly like real pairs."""
    split = fit_split(cfg)
    s1, cand = load(cfg, split, ["entity_id", "country"])
    pairs = candidates(cfg, split)
    ps, pcand = pairs["s1"].to_numpy(), pairs["c"].to_numpy()
    y, _ = labels(cfg, s1["entity_id"], cand["entity_id"], ps, pcand)
    role = split_roles(cfg, s1["entity_id"].to_numpy(zero_copy_only=False))[ps]
    X = model_view(cfg, pair_features(cfg, split))
    tr = np.isin(role, roles)
    if countries:
        tr &= np.isin(s1["country"].to_numpy(zero_copy_only=False)[ps], countries)
    Xtr, ytr, gtr = (X if tr.all() else X[tr]), y[tr], ps[tr].astype(np.int64)     # no copy when all pairs train
    if extra is not None:
        Xtr = pd.concat([Xtr, model_view(cfg, extra["X"])], ignore_index=True)
        ytr, gtr = np.r_[ytr, extra["y"]], np.r_[gtr, extra["s1"].astype(np.int64) + 10**9]
    mdir = cfg.paths.work_dir / f"models{tag}"
    mdir.mkdir(exist_ok=True)
    with stage(f"[4/6] stage-1 model{tag} (pairwise evidence, out-of-fold)"):
        p1tr = model.oof(Xtr, ytr, gtr, cfg)
        m1 = model.fit(Xtr, ytr, cfg)
        p1 = np.empty(len(y), np.float32)
        if (~tr).any():
            p1[~tr] = m1.predict(X[~tr])
        p1[tr] = p1tr[:tr.sum()]
        m1.save_model(mdir / "stage1.txt")
    n_tr, three = int(tr.sum()), cfg.model.stages >= 3
    ctab = coherence_table(cfg, split)
    etab = extra["table"] if extra is not None and cfg.model.coherence else None
    with stage(f"[5/6] stage-2 model{tag} (+ competition features)"):
        X2 = stage2_matrix(X, ps, pcand, p1, ctab)
        del X, Xtr
        X2tr = X2 if tr.all() else X2[tr]
        X2e = None
        if extra is not None:
            X2e = stage2_matrix(model_view(cfg, extra["X"]), extra["s1"], extra["c"], p1tr[n_tr:], etab)
            X2tr = pd.concat([X2tr, X2e], ignore_index=True)
        p2tr = model.oof(X2tr, ytr, gtr, cfg) if three else None
        m2 = model.fit(X2tr, ytr, cfg)
        del X2tr
        m2.save_model(mdir / "stage2.txt")
        p2 = m2.predict(X2).astype(np.float32)
    log("stage-2 top features: " + ", ".join(f"{k}={v:.0f}" for k, v in model.importance(m2).head(12).items()))
    p = p2
    if three:
        with stage(f"[5b/6] stage-3 model{tag} (rivals re-read from out-of-fold stage-2 scores)"):
            p2_in = p2.copy()
            p2_in[tr] = p2tr[:n_tr]
            X3 = stage3_matrix(X2, ps, pcand, p2_in, ctab)
            del X2
            X3tr = X3 if tr.all() else X3[tr]
            if extra is not None:
                X3tr = pd.concat([X3tr, stage3_matrix(X2e, extra["s1"], extra["c"], p2tr[n_tr:], etab)], ignore_index=True)
            m3 = model.fit(X3tr, ytr, cfg)
            del X3tr
            m3.save_model(mdir / "stage3.txt")
            p = m3.predict(X3).astype(np.float32)
        log("stage-3 top features: " + ", ".join(f"{k}={v:.0f}" for k, v in model.importance(m3).head(12).items()))
    elif (mdir / "stage3.txt").exists():
        (mdir / "stage3.txt").unlink()                   # a stale stage-3 model must not be picked up by scoring
    pd.DataFrame({"p1": p1, "p2": p2, "p": p, "y": y, "role": role}).to_parquet(
        cfg.paths.work_dir / f"scores_train{tag}.parquet")


# ----------------------------------------------------------------------------------------------- validation
def _grid_best(fn, grid, ents, gt, n):
    """Pick the parameter maximising macro F0.5 on the *training* entities (never the evaluation entities)."""
    scores = {g: metrics.per_entity(fn(g), gt, ents, n).F.mean() for g in grid}
    return max(scores, key=scores.get)


ROLE_NAMES = {VALID: "valid", TEST: "test"}
MIN_PROB_GRID = (0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5)


def decision_path(cfg):
    return cfg.paths.work_dir / "decision.json"


def min_prob(cfg) -> float:
    """Frozen decision parameter (chosen on VALID by tune_decision), else the config default."""
    p = decision_path(cfg)
    return json.loads(p.read_text())["min_prob"] if p.exists() else cfg.decision.min_prob


def _train_scores(cfg, tag=""):
    split = fit_split(cfg)
    s1, cand = load(cfg, split, ["entity_id", "country"])
    pairs = candidates(cfg, split)
    ps, pcand = pairs["s1"].to_numpy(), pairs["c"].to_numpy()
    sc = pd.read_parquet(cfg.paths.work_dir / f"scores_train{tag}.parquet")
    y, gt = labels(cfg, s1["entity_id"], cand["entity_id"], ps, pcand)
    erole = split_roles(cfg, s1["entity_id"].to_numpy(zero_copy_only=False))
    return s1, cand, ps, pcand, sc, y, gt, erole


def tune_decision(cfg, tag=""):
    """Choose the probability floor of the expected-F0.5 lists on VALID entities and freeze it to decision.json."""
    s1, cand, ps, pcand, sc, y, gt, erole = _train_scores(cfg, tag)
    ents = np.flatnonzero(erole == VALID)
    p = final_prob(sc)
    own = decide.owner(ps, pcand, p)
    grid = {}
    for t in MIN_PROB_GRID:
        sel = np.zeros(len(ps), bool)
        sel[own] = decide.select(ps[own], p[own], t)
        grid[t] = metrics.per_entity((ps[sel], pcand[sel]), gt, ents, cand.num_rows).F.mean()
    best = max(grid, key=grid.get)
    decision_path(cfg).write_text(json.dumps({"min_prob": best, "valid_macro_F0.5": grid[best],
                                              "grid": {str(k): round(v, 5) for k, v in grid.items()}}, indent=2))
    log("decision floor on VALID: " + ", ".join(f"{k}: {v:.4f}" for k, v in grid.items()) + f" -> frozen at {best}")
    return best


def evaluate(cfg, role=VALID, tag="", countries=None, final_only=False):
    """Score one entity role (VALID for selection, TEST once at the end) with the frozen decision.

    Compares fuzzy baseline -> stage-1 model -> + decision layer -> full competitive system, reports slices,
    and writes a pair-level report (confusion matrix, accuracy, P/R/F1, ROC-AUC, PR-AUC, threshold analysis)."""
    s1, cand, ps, pcand, sc, y, gt, erole = _train_scores(cfg, tag)
    _, cflags = load(cfg, fit_split(cfg), ["f_pseudo", "f_alias", "f_url", "f_script", "f_empty_addr"])
    X = pair_features(cfg, fit_split(cfg))[["nm_tset", "ad_tset", "num_p_in"]]
    n = cand.num_rows
    ectry = s1["country"].to_numpy(zero_copy_only=False)
    hold, tr_e = np.flatnonzero(erole == role), np.flatnonzero(erole == TRAIN)
    if countries:
        hold = hold[np.isin(ectry[hold], countries)]
    in_tr = erole[ps] == TRAIN
    name = ROLE_NAMES[role] + tag
    p1, p2, pf = sc.p1.to_numpy(), sc.p2.to_numpy(), final_prob(sc)

    def pred(mask):
        return ps[mask], pcand[mask]

    def baselines():
        g = [(a, b) for a in (70, 80, 85, 90, 95) for b in (60, 70, 80, 90)]
        ab = _grid_best(lambda t: pred(in_tr & (X.nm_tset.values >= t[0]) & (X.ad_tset.values >= t[1])), g, tr_e, gt, n)
        t1 = _grid_best(lambda t: pred(in_tr & (p1 >= t)), [0.3, 0.4, 0.5, 0.6, 0.7, 0.8], tr_e, gt, n)
        return {
            f"B0 fuzzy rule (name>={ab[0]}, addr>={ab[1]})": (X.nm_tset.values >= ab[0]) & (X.ad_tset.values >= ab[1]),
            f"B1 stage-1 model, global threshold {t1}": p1 >= t1,
            "B2 stage-1 + ownership + expected-F0.5 lists": final_decision(cfg, ps, pcand, p1),
            "B3a stage-2 (competition), threshold 0.5": p2 >= 0.5,
        }

    with stage(f"evaluation ({name})"):
        systems = {} if final_only else baselines()
        systems["FINAL last stage + ownership + expected-F0.5"] = final_sel = final_decision(cfg, ps, pcand, pf,
                                                                                             source3(cand))
        per = {k: metrics.per_entity(pred(m), gt, hold, n) for k, m in systems.items()}

    # entity-level slices (labels used only to *describe* entities, never to predict)
    gt_df = pd.DataFrame({"s": gt[0], "c": gt[1]})
    cf = {f: cflags[f].to_numpy(zero_copy_only=False) for f in cflags.column_names}
    unusual = gt_df[(cf["f_pseudo"] | cf["f_alias"] | cf["f_url"] | cf["f_script"])[gt_df.c]].s.unique()
    empty = gt_df[cf["f_empty_addr"][gt_df.c]].s.unique()
    lookalike = np.unique(ps[(y == 0) & (X.nm_tset.values >= SIB_NAME)])
    n_c = pd.Series(np.bincount(ps, minlength=s1.num_rows))
    slices = {
        "country": pd.Series(ectry),
        "singleton": pd.Series(np.bincount(gt[0], minlength=s1.num_rows) == 0),
        "has look-alike decoy candidate": pd.Series(np.isin(np.arange(s1.num_rows), lookalike)),
        "has unusual-name true match": pd.Series(np.isin(np.arange(s1.num_rows), unusual)),
        "has empty-address true match": pd.Series(np.isin(np.arange(s1.num_rows), empty)),
        "candidates": pd.cut(n_c, [-1, 0, 3, 6, 10, 20, 10**6], labels=["0", "1-3", "4-6", "7-10", "11-20", ">20"]).astype(str),
    }
    comp = pd.DataFrame({k: metrics.report(d, slices).loc[:, "F0.5"] for k, d in per.items()}).T
    final = metrics.report(per[list(per)[-1]], slices)

    in_eval, n_gt = np.isin(ps, hold), int(np.isin(gt[0], hold).sum())
    rep = metrics.pair_report(y[in_eval], final_sel[in_eval], pf[in_eval], n_gt)
    rep.update({"entities": int(len(hold)), "macro_F0.5": float(per[list(per)[-1]].F.mean()),
                "oracle_macro_F0.5": float(metrics.per_entity(pred(y == 1), gt, hold, n).F.mean()),
                "decision_min_prob": min_prob(cfg)})
    thr = metrics.threshold_table(y[in_eval], pf[in_eval], n_gt)
    own = decide.owner(ps, pcand, pf)
    thr["macro_F0.5 (owner + threshold)"] = [metrics.per_entity(pred(own & (pf >= t)), gt, hold, n).F.mean()
                                             for t in thr.index]
    out = cfg.paths.work_dir / "validation"
    out.mkdir(exist_ok=True)
    comp.to_csv(out / f"comparison_{name}.csv")
    final.to_csv(out / f"final_slices_{name}.csv")
    thr.to_csv(out / f"thresholds_{name}.csv")
    (out / f"report_{name}.json").write_text(json.dumps(rep, indent=2))
    pd.set_option("display.width", 250)
    log(f"{name} entities: {len(hold):,}\nF0.5 by system and slice:\n{comp.round(4).T.to_string()}\n\n"
        f"FINAL system detail:\n{final.round(4).to_string()}\n\npair-level report:\n{json.dumps(rep, indent=2)}\n\n"
        f"threshold analysis (candidate pairs):\n{thr.round(4).to_string()}")
    return comp, final


def validate(cfg, tag="", countries=None, final_only=False):
    return evaluate(cfg, VALID, tag, countries, final_only)


# ----------------------------------------------------------------------------------------------- inference
UNLABELLED = "_unlabelled"   # model tag for countries without training labels (trained with the synthetic block)
FINAL = "_final"             # models refit on every labelled entity once the design is frozen
LABELLED_ROLES = (UNUSED, TRAIN, VALID, TEST)


def labelled_countries(cfg) -> list:
    s1, _ = io.load_split(cfg, "train", get_lexicon(cfg), ["country"])
    return sorted(set(s1["country"].to_numpy(zero_copy_only=False)))


def boosters(cfg, tag=""):
    import lightgbm as lgb
    d = cfg.paths.work_dir / f"models{tag}"
    if not (d / "stage2.txt").exists():
        return None
    return tuple(lgb.Booster(model_file=str(d / f"stage{k}.txt")) for k in (1, 2, 3) if (d / f"stage{k}.txt").exists())


def _predict(model, X: pd.DataFrame) -> np.ndarray:
    assert model.feature_name() == list(X.columns), "feature order mismatch between model and feature store"
    return model.predict(X.to_numpy(np.float32)).astype(np.float32)


def _run(models, X, ps, pcand, ctab):
    """(stage-1 probability, last-stage probability, last feature matrix) of every pair under one model set."""
    p1 = _predict(models[0], X)
    Xs = stage2_matrix(X, ps, pcand, p1, ctab)
    p = _predict(models[1], Xs)
    if len(models) > 2:
        Xs = stage3_matrix(Xs, ps, pcand, p, ctab)
        p = _predict(models[2], Xs)
    return p1, p, Xs


def score(cfg, split):
    """(pairs, last feature matrix, p1, p) for a split.

    Labelled countries are scored by the models trained on real labels. Countries without training labels get
    w * (synthetic/self-trained model) + (1 - w) * (labelled model), w = synth.unlabelled_weight. Candidate pairs
    never cross countries, so each country's competition features are complete under either model set."""
    pairs = candidates(cfg, split)
    ps, pcand = pairs["s1"].to_numpy(), pairs["c"].to_numpy()
    s1, _ = load(cfg, split, ["country"])
    route = ~np.isin(s1["country"].to_numpy(zero_copy_only=False)[ps], labelled_countries(cfg))
    base, unl = boosters(cfg, FINAL) or boosters(cfg), boosters(cfg, FINAL + UNLABELLED) or boosters(cfg, UNLABELLED)
    w = float(getattr(cfg.synth, "unlabelled_weight", 1.0)) if unl is not None else 0.0
    ctab = coherence_table(cfg, split)
    X = model_view(cfg, pair_features(cfg, split))
    p1, p, Xs = _run(base, X, ps, pcand, ctab)
    if route.any() and w > 0:
        r = np.flatnonzero(route)
        _, pu, _ = _run(unl, X.iloc[r].reset_index(drop=True), ps[r], pcand[r], ctab)
        p[r] = w * pu + (1 - w) * p[r]
    log(f"{split}: {(~route).sum():,} pairs scored by the labelled-country model, {route.sum():,} unlabelled-country "
        f"pairs by {w:.2f} x unlabelled model + {1 - w:.2f} x labelled model")
    return ps, pcand, Xs, p1, p


def source3(cand) -> np.ndarray:
    """Per candidate record: True for Source 3, False for Source 2 (from the id prefix)."""
    return pc.starts_with(cand["entity_id"], "S3").to_numpy(zero_copy_only=False)


def final_decision(cfg, ps, pcand, p, src3=None):
    """Ownership + expected-F0.5 lists, then (when src3 is given) the cross-source fill."""
    own = decide.owner(ps, pcand, p)
    sel = np.zeros(len(p), bool)
    sel[own] = decide.select(ps[own], p[own], min_prob(cfg))
    t = getattr(cfg.decision, "cross_source_min_prob", 0.0)
    if src3 is not None and t > 0:
        sel = decide.cross_source_fill(ps, pcand, p, sel, src3, t)
    return sel


def predict(cfg):
    s1, cand = load(cfg, "test", ["entity_id", "country"])
    with stage("[5/6] scoring test candidates"):
        ps, pcand, X2, p1, p2 = score(cfg, "test")
    with stage("[6/6] ownership + expected-F0.5 match lists"):
        sel = final_decision(cfg, ps, pcand, p2, source3(cand))
    s1_ids, c_ids = s1["entity_id"].to_numpy(zero_copy_only=False), cand["entity_id"].to_numpy(zero_copy_only=False)
    out = cfg.paths.output_dir
    io.write_lists(out / "candidate_pairs.tsv", s1_ids, ps, pcand, c_ids, ["source1_entity_id", "candidate_entity_ids"])
    io.write_lists(out / "matching_results.tsv", s1_ids, ps[sel], pcand[sel], c_ids, ["source1_entity_id", "matched_entity_ids"])
    ctry = s1["country"].to_numpy(zero_copy_only=False)
    per = pd.DataFrame({"country": ctry, "n": np.bincount(ps[sel], minlength=len(ctry))})
    log("test predictions per country:\n" + per.groupby("country").n.agg(entities="size", matches="sum",
        empty=lambda x: (x == 0).mean(), mean="mean").round(3).to_string())
    pd.DataFrame({"p1": p1, "p2": p2, "sel": sel}).to_parquet(cfg.paths.work_dir / "scores_test.parquet")
