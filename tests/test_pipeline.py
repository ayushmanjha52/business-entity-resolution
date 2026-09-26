"""End-to-end integration test on a tiny generated dataset in the exact challenge TSV format.

prepare -> lexicon -> blocking -> stage-1/2 training -> validation -> test prediction -> official validator.
The test split contains a country absent from training (open-set country handling)."""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import pandas as pd

from doppelganger import config, pipeline, synth
from doppelganger.__main__ import _final

ROOT = Path(__file__).resolve().parents[1]
STREETS = ["Oak", "Pine", "Maple", "Cedar", "Elm", "Birch", "Walnut", "Spruce", "Willow", "Aspen"]
WORDS = ["Summit", "Harbor", "Vertex", "Crescent", "Atlas", "Nimbus", "Quartz", "Falcon", "Meridian", "Beacon",
         "Juniper", "Orion", "Sable", "Tundra", "Cobalt", "Ember", "Lumen", "Prairie", "Radiant", "Solace"]
KINDS = ["Dental", "Logistics", "Bakery", "Consulting", "Clinic", "Holdings", "Labs", "Foods"]


def _entities(n, country, city, region, rng):
    rows = []
    for i in range(n):
        name = f"{rng.choice(WORDS)} {rng.choice(WORDS)} {rng.choice(KINDS)} {rng.choice(['LLC', 'Inc', 'Ltd'])}"
        rows.append((name, f"{rng.integers(10, 9999)} {rng.choice(STREETS)} Street, {city}, {region}", country))
    return pd.DataFrame(rows, columns=["business_name", "business_address", "country"])


def _profile():
    rates = dict(pseudo=0.02, alias=0.02, url=0.05, junk=0.02, upper_name=0.2, reorder=0.1, drop=0.1, add=0.1,
                 typo=0.2, accent=0.05, empty_addr=0.04, upper_addr=0.4, comp_reorder=0.05, comp_drop=0.05,
                 num_missing=0.05, num_inject=0.1, addr_typo=0.1)
    return synth.Profile(rates=rates, n_variants=[0, 1, 2, 3, 3, 4], decoy_delta=[2, 4, 7, 9, 13], decoys_per_entity=1.0,
                         descriptors=["north", "center", "partners"], legal=["llc", "inc"], pseudo=["Zorvexa", "Quilmora"],
                         markers=["aka", "DBA"], num_prefix=["#", "No"], region_variants={}, abbrev={"st": ["St", "ST"]})


def _write_split(d: Path, split, s1, gen):
    s1 = s1.reset_index(drop=True)
    s1.insert(0, "entity_id", [f"S1-{split[:2]}{i}" for i in range(len(s1))])
    recs = gen.generate(s1, len(s1), prefix=split.upper())
    (d / split).mkdir(parents=True, exist_ok=True)
    s1[["entity_id", "business_name", "business_address", "country"]].to_csv(d / split / f"{split}_source1.tsv", sep="\t", index=False)
    for k in (2, 3):
        recs[recs.entity_id.str.startswith(f"S{k}-")][["entity_id", "business_name", "business_address", "country"]] \
            .to_csv(d / split / f"{split}_source{k}.tsv", sep="\t", index=False)
    if split == "train":
        true = recs[~recs.decoy]
        lists = true.groupby("origin").entity_id.agg(",".join)
        gt = pd.DataFrame({"source1_entity_id": s1.entity_id, "matched_entity_ids": lists.reindex(s1.index).fillna("").values})
        gt.to_csv(d / "train" / "train_ground_truth.tsv", sep="\t", index=False)


@pytest.mark.parametrize("augment", [False, True])
def test_end_to_end(tmp_path, monkeypatch, augment):
    rng = np.random.default_rng(1)
    gen = synth.Generator(_profile(), 3)
    data = tmp_path / "dataset"
    _write_split(data, "train", pd.concat([_entities(700, "US", "Tulsa", "OK", rng),
                                           _entities(500, "India", "Pune", "Maharashtra", rng)]), gen)
    _write_split(data, "test", pd.concat([_entities(150, "US", "Tulsa", "OK", rng),
                                          _entities(120, "Spain", "Madrid", "Madrid", rng)]), gen)   # unseen country
    cfg = config.load()
    cfg.paths.data_dir, cfg.paths.work_dir, cfg.paths.output_dir = data, tmp_path / "work", tmp_path / "output"
    cfg.paths.work_dir.mkdir()
    cfg.split.valid_frac, cfg.split.test_frac, cfg.split.train_frac = 0.2, 0.2, 0.6
    cfg.model.n_estimators, cfg.model.min_child_samples = 60, 10
    cfg.lexicon.component_min_count = 5
    cfg.split.decoy_augment = augment
    if augment:                                                  # fixture splits share one density: force +1 decoy
        from doppelganger import augment as aug
        monkeypatch.setattr(aug, "extra_decoys_per_entity", lambda cfg: 1.0)

    pipeline.blocking_report(cfg)
    pipeline.train(cfg)
    pipeline.tune_decision(cfg)
    comp, final = pipeline.validate(cfg)
    _, test = pipeline.evaluate(cfg, pipeline.TEST)
    assert test.loc["ALL", "F0.5"] > 0.8                        # unseen TEST entities, frozen decision
    assert final.loc["ALL", "F0.5"] > 0.8                       # sanity: the model learned something real
    assert comp.loc[comp.index[-1], "ALL"] >= comp.loc[comp.index[0], "ALL"]   # final >= fuzzy baseline
    _final(cfg)                                                 # refit + synthetic + self-training round
    pipeline.predict(cfg)

    out = cfg.paths.output_dir
    m = pd.read_csv(out / "matching_results.tsv", sep="\t", dtype=str, keep_default_na=False)
    c = pd.read_csv(out / "candidate_pairs.tsv", sep="\t", dtype=str, keep_default_na=False)
    s1 = pd.read_csv(data / "test" / "test_source1.tsv", sep="\t", dtype=str)
    assert list(m.columns) == ["source1_entity_id", "matched_entity_ids"]
    assert list(c.columns) == ["source1_entity_id", "candidate_entity_ids"]
    assert m.source1_entity_id.tolist() == s1.entity_id.tolist()               # every entity exactly once
    for mm, cc in zip(m.matched_entity_ids, c.candidate_entity_ids):
        ids = [x for x in mm.split(",") if x]
        assert len(ids) == len(set(ids)) and set(ids) <= set(cc.split(","))    # no dups, matches ⊆ candidates
        assert all(x[:3] in ("S2-", "S3-") for x in ids)
    spain = m[s1.country == "Spain"]
    assert (spain.matched_entity_ids != "").mean() > 0.5                       # open-set country still matched
    validator = ROOT / "student_resource" / "utils" / "validate_submission.py"
    if not validator.exists():                                                 # packaged copy ships without student_resource
        return
    r = subprocess.run([sys.executable, str(validator),
                        "--matching", str(out / "matching_results.tsv"), "--candidate", str(out / "candidate_pairs.tsv"),
                        "--test-dir", str(data / "test"), "--check-ids"], capture_output=True, text=True)
    assert r.returncode == 0 and "PASS" in r.stdout, r.stdout + r.stderr
