import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import shap
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

ROOT = Path(__file__).parent
M = ROOT / "models"
META = json.load(open(M / "feature_meta.json"))
EXAMPLES = json.load(open(M / "examples.json"))
COLUMNS = joblib.load(M / "feature_columns.joblib")
TARGETS = ["CAD", "LAD", "LCX", "RCA"]
MODELS = {t: joblib.load(M / f"{t}.joblib") for t in TARGETS}
EXPLAINERS = {t: shap.TreeExplainer(m) for t, m in MODELS.items()}
TOP_N = 8

app = FastAPI(title="Cardiac risk API (decision support / education only)")

class Patient(BaseModel):
    features: dict[str, float]

def explain(t, X):
    sv = EXPLAINERS[t](X, check_additivity=False)
    vals = np.asarray(sv.values)
    vals = vals[0, :, 1] if vals.ndim == 3 else vals[0]
    base = np.asarray(sv.base_values)
    base = float(base[0, 1]) if base.ndim == 2 else float(np.ravel(base)[0])
    order = np.argsort(-np.abs(vals))[:TOP_N]
    return {"base": base,
            "top": [{"feature": COLUMNS[i], "value": float(X.iloc[0, i]),
                     "contribution": float(vals[i])} for i in order]}

@app.get("/api/meta")
def meta():
    return {**META, "examples": EXAMPLES}

@app.post("/api/predict")
def predict(p: Patient):
    unknown = [k for k in p.features if k not in COLUMNS]
    if unknown:
        raise HTTPException(422, f"Unknown features: {unknown}")
    row = {**META["defaults"], **p.features}
    X = pd.DataFrame([row])[COLUMNS]
    out = {t: float(MODELS[t].predict_proba(X)[0, 1]) for t in TARGETS}
    out["explanations"] = {t: explain(t, X) for t in TARGETS}
    return out

app.mount("/", StaticFiles(directory=ROOT / "web", html=True), name="web")
