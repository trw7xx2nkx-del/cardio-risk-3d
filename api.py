import json
from pathlib import Path
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

ROOT = Path(__file__).parent
M = ROOT / "models"
META = json.load(open(M / "feature_meta.json"))
EXAMPLES = json.load(open(M / "examples.json"))
COLUMNS = joblib.load(M / "feature_columns.joblib")
MODELS = {t: joblib.load(M / f"{t}.joblib") for t in ["CAD", "LAD", "LCX", "RCA"]}

app = FastAPI(title="Cardiac risk API (decision support / education only)")

class Patient(BaseModel):
    features: dict[str, float]

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
    return {t: float(m.predict_proba(X)[0, 1]) for t, m in MODELS.items()}

app.mount("/", StaticFiles(directory=ROOT / "web", html=True), name="web")
