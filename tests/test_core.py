"""Focused unit tests on tiny fixtures (the full 22M-record pipeline is never run here)."""
import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

from doppelganger import blocking, decide, features, metrics, normalize as nz
from doppelganger.lexicon import Lexicon, learn_components

CFG = type("C", (), {})()
CFG.blocking = type("B", (), dict(s1_key_cap=30, top_per_record=8, min_rel_votes=0.0, chunk_size=1000, number_keys=3))()
CFG.prep = type("P", (), dict(max_numbers=6))()


def raw(rows):
    return pa.table({k: [r[i] for r in rows] for i, k in enumerate(["entity_id", "business_name", "business_address", "country"])})


LEX = Lexicon(translit={"लिमिटेड": "limited", "बेस्ट": "best", "इंफोटेक": "infotech"},
              components={"India|tn": "tamil nadu", "France|nord": "hauts de france"},
              localities=["India|tamil nadu", "France|hauts de france"],
              vocab=["best", "infotech", "keystone", "logistics", "international", "vanguard", "star", "iron"],
              pseudo_min_len=5)


def prep(rows):
    return nz.finalize(nz.base_normalize(raw(rows), 6), LEX).to_pandas()


# ---------------------------------------------------------------- normalisation
def test_name_normalisation_and_noise_flags():
    d = prep([("S2-1", "Evonex t/a Vanguard Star Iron", "", "US"),
              ("S2-2", "-- KEYST0NE L.L.C. & Co (ID: 123)", "1 A St", "US"),
              ("S2-3", "orthopedicsafehealth.com", "1 A St", "US"),
              ("S2-4", "Onyxonyx", "1 A St", "US"),
              ("S2-5", "Café Fócus SARL | www.x.com", "1 A St", "France")])
    assert d.name_core[0] == "vanguard star iron" and d.f_alias[0]          # alias -> real name kept
    assert d.f_junk[1] and d.name_full[1] == "keystone llc and co"           # junk prefix, homoglyph, dotted legal suffix
    assert d.f_url[2] and d.name_nospace[2] == "orthopedicsafehealth"
    assert d.f_pseudo[3] and not d.f_pseudo[2]                                # pseudo-name, but URL is not pseudo
    assert d.name_core[4] == "cafe focus"                                     # accents folded, legal form removed


def test_native_script_tokens_survive_and_transliterate():
    d = prep([("S2-1", "बेस्ट इंफोटेक लिमिटेड", "Mumbai", "India")])
    assert d.name_full[0] == "best infotech limited" and d.name_core[0] == "best infotech"
    assert d.f_script[0]


def test_house_number_extraction():
    t = nz.base_normalize(raw([("S2-1", "x", "#7-67 Plot No: 051, PO Box 5619, Chennai", "India"),
                               ("S2-2", "x", "", "US")]), 6).to_pandas()
    assert [t.num0[0], t.num1[0], t.num2[0], t.num3[0]] == [7, 67, 51, -1]   # leading zero folded, PO box removed
    assert t.n_nums[1] == 0 and t.f_empty_addr[1] and t.num0[1] == -1


def test_region_equivalence_and_abbreviations():
    d = prep([("S3-1", "x", "22 Rue Racine, Dunkerque, Nord", "France"),
              ("S1-1", "x", "22 R. Racine, Dunkerque, Hauts-de-France", "France"),
              ("S3-2", "x", "5 Main Street, Chennai, TN", "India")])
    assert d.addr_norm[0] == d.addr_norm[1] == "22 rue racine dunkerque hauts france"
    assert list(d.localities[0]) == ["France|hauts de france"]
    assert d.addr_norm[2].endswith("tamil nadu") and "st" in d.addr_norm[2].split()


def test_country_is_open_set():
    d = prep([("S2-1", "Acme", "1 Calle Mayor, Madrid", "Spain")])       # unseen country passes through
    assert d.country[0] == "Spain" and d.addr_norm[0] == "1 calle mayor madrid"


def test_learn_components_maps_variant_to_canonical():
    s1 = nz.base_normalize(raw([("S1-%d" % i, "x", f"{i} A Rd, Tamil Nadu", "India") for i in range(60)]), 6)
    c = nz.base_normalize(raw([("S2-%d" % i, "x", f"{i} A Rd, TN", "India") for i in range(60)]), 6)
    a = np.arange(60)
    s1_rate = pd.Series({"India|tamil nadu": 1.0})          # "tn" never appears in Source 1 -> foreign variant
    c_rate = pd.Series({"India|tn": 1.0})
    cfg = type("L", (), dict(component_min_share=0.5, component_min_count=40, foreign_max_ratio=0.2))()
    m = learn_components(s1, c, a, a, s1_rate, c_rate, cfg)
    assert m == {"India|tn": "tamil nadu"}


# ---------------------------------------------------------------- blocking
def test_blocking_finds_noisy_match_and_separates_branches():
    s1 = prep([("S1-1", "Serra Delta Biotechnologies", "1619 Pine St N, Little Rock, AR", "US"),
               ("S1-2", "Serra Delta Biotechnologies", "88 Oak Ave, Tulsa, OK", "US"),
               ("S1-3", "Unrelated Bakery", "5 Elm St, Tulsa, OK", "US")])
    c = prep([("S2-1", "SERRA DELTA BIOTECHNOLOGIES", "PINE ST N, LITTLE ROCK, AR", "US"),   # no house number
              ("S2-2", "Onyxonyx", "1619 Pine Street North, Little Rock, AR", "US")])         # pseudo-name
    s1t, ct = pa.Table.from_pandas(s1), pa.Table.from_pandas(c)
    pairs = blocking.generate(s1t, ct, CFG)
    best = pd.DataFrame(pairs).sort_values("votes", ascending=False).drop_duplicates("c").set_index("c").s1
    assert best[0] == 0 and best[1] == 0                   # both reach the Little Rock branch first


# ---------------------------------------------------------------- features
def test_number_geometry_distinguishes_decoy_from_variant():
    n1 = np.array([[668, -1], [1681, -1], [215, -1]])
    n2 = np.array([[661, -1], [7, 1681], [15, -1]])        # decoy shift | injected door no. | dropped digit
    f = features.number_features(n1, n2)
    assert f["num_p_in"].tolist() == [0, 1, 0]
    assert f["num_logdiff"][0] == pytest.approx(np.log1p(7))
    assert f["num_drop"].tolist() == [0, 0, 1]
    assert np.isnan(features.number_features(np.array([[5, -1]]), np.array([[-1, -1]]))["num_p_in"][0])


def test_competition_features_margin():
    comp = features.competition_features(np.array([0, 1, 0]), np.array([10, 10, 11]),
                                         np.array([0.9, 0.3, 0.2], np.float32), np.zeros(3, np.float32))
    assert comp.c_margin.tolist() == pytest.approx([0.6, -0.6, 0.2])
    assert comp.e_rank.tolist() == [1, 1, 2]


# ---------------------------------------------------------------- decisions & metric
def test_owner_keeps_one_entity_per_record():
    keep = decide.owner(np.array([0, 1, 2]), np.array([7, 7, 8]), np.array([0.4, 0.9, 0.1]))
    assert keep.tolist() == [False, True, True]


def test_select_expected_f05():
    s1 = np.array([0, 0, 0, 1, 2, 2])
    p = np.array([0.95, 0.9, 0.1, 0.2, 0.6, 0.55])
    sel = decide.select(s1, p)
    assert sel.tolist() == [True, True, False, False, True, True]   # drops the 0.1 tail; entity 1 stays empty


def test_metric_matches_readme_example():
    d = metrics.per_entity((np.array([0, 0, 0]), np.array([47, 193, 812])), (np.array([0, 0]), np.array([47, 812])),
                           np.array([0, 1]), 1000)
    assert d.F[0] == pytest.approx(0.714, abs=1e-3)
    assert d.F[1] == 1.0                                      # singleton predicted empty scores 1


def test_coherence_features_use_the_entitys_most_confident_other_record():
    cand = pa.table({"name_core": ["acme tools", "acme tools", "zeta", "other co"],
                     "addr_alpha": ["pine st tulsa", "", "pine st tulsa", "elm ave"],
                     "num0": [12, -1, 12, 9], "num1": [-1] * 4, "num2": [-1] * 4})
    s1 = np.array([0, 0, 0, 1])
    c = np.array([0, 1, 2, 3])
    p1 = np.array([0.95, 0.30, 0.60, 0.8], np.float32)
    f = features.coherence_features(s1, c, p1, cand)
    assert f.coh_p.tolist()[:3] == pytest.approx([0.60, 0.95, 0.95])      # anchor = best *other* record
    assert f.coh_nm[1] == 100                                             # empty-address record: same name as anchor
    assert f.coh_num[2] == 1 and np.isnan(f.coh_num[1])                   # shares number 12 / no number to compare
    assert np.isnan(f.coh_p[3])                                           # single-candidate entity has no anchor


# ---------------------------------------------------------------- v3: noise-inversion helpers
def test_url_names_are_segmented_into_vocabulary_words():
    lex = Lexicon(translit={}, components={}, localities=[], vocab=["impex", "swastik", "center"], pseudo_min_len=5)
    d = nz.finalize(nz.base_normalize(raw([("S2-1", "impexprivate.com", "", "India"),
                                           ("S2-2", "zzqxv.com", "", "India")]), 6), lex).to_pandas()
    assert list(d.name_tokens[0]) == ["impex"] and d.name_nospace[0] == "impexprivate"   # legal word dropped
    assert list(d.name_tokens[1]) == ["zzqxv"]                                             # unexplained: kept


def test_homoglyph_folding_only_touches_mixed_tokens():
    out = nz.fold_homoglyphs(pa.array(["c0mmerce", "samue1", "6cs", "1062", "abc"])).to_pylist()
    assert out == ["commerce", "samuel", "gcs", "1062", "abc"]


def test_digit_deletions_cover_the_observed_number_noise():
    d = blocking.digit_deletions(np.array([2677, 7, -1]))
    assert 677 in d[0] and 267 in d[0] and (d[1] == -1).all() and (d[2] == -1).all()


def test_rarity_overlap_prefers_rare_shared_tokens():
    s1 = pa.table({"country": ["US"] * 3, "name_core": ["acme", "acme", "zeta rare"],
                   "name_tokens": [["acme"], ["acme"], ["zeta", "rare"]], "addr_tokens": [["main"], ["main"], ["oak"]]})
    r = features.TokenRarity(s1)
    t1 = s1.take([2, 2])
    t2 = pa.table({"country": ["US"] * 2, "name_core": ["zeta", "rare x"],
                   "name_tokens": [["zeta"], ["rare", "x"]], "addr_tokens": [["oak"], []]})
    o = r.overlap("name_tokens", t1, t2, "nm")
    assert o["nm_cov1"][0] == pytest.approx(0.5) and o["nm_cov2"][0] == pytest.approx(1.0)
    assert o["nm_cov2"][1] < 1.0                                         # unknown cand token counts against it
    assert list(r.name_count(s1)) == [2, 2, 1]
