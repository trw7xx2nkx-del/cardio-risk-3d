from fastapi.testclient import TestClient
import api

client = TestClient(api.app)
TARGETS = ["CAD", "LAD", "LCX", "RCA"]

def test_meta_lists_features_and_reliability():
    m = client.get("/api/meta").json()
    assert "Age" in m["columns"] and "examples" in m
    assert set(TARGETS) <= set(m["reliability"])

def test_predict_returns_valid_probabilities_and_explanations():
    r = client.post("/api/predict", json={"features": {"Age": 60}})
    assert r.status_code == 200
    d = r.json()
    for t in TARGETS:
        assert 0.0 <= d[t] <= 1.0
        e = d["explanations"][t]
        assert e["text"]["summary"].startswith("Estimated probability")
        assert len(e["top"]) == 8

def test_unknown_feature_is_rejected():
    assert client.post("/api/predict", json={"features": {"Nope": 1}}).status_code == 422

def test_contributions_add_up_to_prediction():
    d = client.post("/api/predict", json={"features": {"Age": 55, "HTN": 1}}).json()
    for t in TARGETS:
        e = d["explanations"][t]
        assert abs(e["base"] + sum(e["categories"].values()) - d[t]) < 0.03

def test_typical_chest_pain_raises_cad_probability():
    low = client.post("/api/predict", json={"features": {"Typical Chest Pain": 0}}).json()["CAD"]
    high = client.post("/api/predict", json={"features": {"Typical Chest Pain": 1}}).json()["CAD"]
    assert high > low
