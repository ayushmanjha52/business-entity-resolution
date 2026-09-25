"""Explorer API on a tiny evidence store (no full pipeline needed)."""
import lightgbm as lgb
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from doppelganger import config
from doppelganger.explorer.app import create_app

FEATS = ["nm_tset", "ad_tset", "num_p_in", "num_logdiff", "num_drop", "loc_overlap", "c_f_pseudo", "c_f_alias",
         "c_f_url", "c_f_script", "c_f_empty_addr", "e_sib", "e_rank", "num_frac1", "tok_diff"]


def test_explorer_endpoints(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.random((200, len(FEATS))), columns=FEATS)
    b = lgb.train({"objective": "binary", "verbose": -1}, lgb.Dataset(X, (X.nm_tset > 0.5).astype(int)), 5)
    (tmp_path / "models").mkdir()
    b.save_model(str(tmp_path / "models" / "stage2.txt"))
    ev = pd.DataFrame({
        "s1_id": ["S1-1"] * 3, "s1_name": ["Lockport Advanced Telefonica Inc"] * 3,
        "s1_addr": ["661 Locust Street, Lockport, NY"] * 3, "country": "US", "split": "train",
        "c_id": ["S2-1", "S2-2", "S3-3"], "c_name": ["LOCKPORT ADVANCED TELEFONICA", "Lockport Advanced Telefonica Corp", "Onyxonyx"],
        "c_addr": ["661 LOCUST ST, LOCKPORT, NY", "668 LOCUST ST, LOCKPORT, NY", "661 Locust St, Lockport, New York"],
        "p1": [0.99, 0.93, 0.40], "p2": [0.99, 0.04, 0.92], "selected": [True, False, True],
        "label": [1, 0, 1], "rival_id": ["", "", ""]})
    fx = pd.DataFrame({"nm_tset": [100, 95, 10], "ad_tset": [100, 100, 100], "num_p_in": [1, 0, 1],
                       "num_logdiff": [0, np.log1p(7), 0], "num_drop": [0, 0, 0], "loc_overlap": [2, 2, 2],
                       "c_f_pseudo": [0, 0, 1], "c_f_alias": 0, "c_f_url": 0, "c_f_script": 0, "c_f_empty_addr": 0,
                       "e_sib": [1, 1, 1], "e_rank": [1, 3, 2], "num_frac1": [1, 0, 1],
                       "tok_diff": [0, 1, -2]}, dtype=float)
    ev.to_parquet(tmp_path / "evidence.parquet")
    fx.to_parquet(tmp_path / "evidence_features.parquet")
    cfg = config.load()
    cfg.paths.work_dir = tmp_path
    c = TestClient(create_app(cfg))

    assert c.get("/api/demo/lookalike").json()["s1_id"] == "S1-1"
    assert c.get("/api/demo/lowsim").json()["s1_id"] == "S1-1"
    d = c.get("/api/entity/S1-1").json()
    assert [x["verdict"] for x in d["candidates"]] == ["matched", "matched", "rejected"]
    decoy = d["candidates"][-1]
    assert any("661 absent; record has 668 instead" in r[1] for r in decoy["reasons"])
    assert any("descriptor" in r[1] for r in decoy["reasons"])
    assert not any("wins over" in r[1] for r in decoy["reasons"])           # rejected records get no win reasons
    assert d["fooled"] == 1 and d["correct"] == 3
    assert len(decoy["shap"]) == 6
    assert c.get("/api/search", params={"q": "lockport"}).json()[0]["s1_id"] == "S1-1"
    assert c.get("/api/entity/S1-404").status_code == 404
    assert c.get("/api/demo/nope").status_code == 404
    assert c.get("/api/search", params={"q": "x"}).status_code == 422      # input validation
    assert "Doppelgänger" in c.get("/").text
