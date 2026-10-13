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
CAL = json.load(open(M / "calibration.json")) if (M / "calibration.json").exists() else {}
TOP_N = 8

RELIABILITY = {}
metrics_file = ROOT / "docs" / "final_metrics.csv"
if metrics_file.exists():
    for _, r in pd.read_csv(metrics_file).iterrows():
        RELIABILITY[r["target"]] = {"auc": float(r["roc_auc"]), "lo": float(r["auc_ci_low"]),
                                    "hi": float(r["auc_ci_high"])}

CATEGORY = {}
for cat, cols in {
    "Demographics": ["Age", "Sex", "Weight", "Length", "BMI"],
    "Risk factors and history": ["DM", "HTN", "Current Smoker", "EX-Smoker", "FH", "Obesity",
                                 "CRF", "CVA", "Airway disease", "Thyroid Disease", "CHF", "DLP"],
    "Exam findings": ["BP", "PR", "Edema", "Weak Peripheral Pulse", "Lung rales",
                      "Systolic Murmur", "Diastolic Murmur"],
    "Symptoms": ["Typical Chest Pain", "Dyspnea", "Function Class", "Atypical", "Nonanginal", "LowTH Ang"],
    "ECG": ["Q Wave", "St Elevation", "St Depression", "Tinversion", "LVH",
            "Poor R Progression", "BBB_LBBB", "BBB_RBBB"],
    "Echocardiography": ["EF-TTE", "Region RWMA", "VHD"],
    "Blood tests": ["FBS", "CR", "TG", "LDL", "HDL", "BUN", "ESR", "HB", "K", "Na",
                    "WBC", "Lymph", "Neut", "PLT"],
}.items():
    for c in cols:
        CATEGORY[c] = cat

app = FastAPI(title="Cardiac risk API (decision support / education only)")

class Patient(BaseModel):
    features: dict[str, float]

def calibrate(t, p):
    if t not in CAL:
        return float(p)
    q = float(np.clip(p, 1e-3, 1 - 1e-3))
    return float(1 / (1 + np.exp(-(CAL[t]["a"] * np.log(q / (1 - q)) + CAL[t]["b"]))))

def explain(t, X, raw_p):
    sv = EXPLAINERS[t](X, check_additivity=False)
    vals = np.asarray(sv.values)
    vals = vals[0, :, 1] if vals.ndim == 3 else vals[0]
    base = np.asarray(sv.base_values)
    base = float(base[0, 1]) if base.ndim == 2 else float(np.ravel(base)[0])
    # Rescale so the bars add up to (calibrated prediction - calibrated baseline)
    if abs(raw_p - base) > 1e-3:
        scale = (calibrate(t, raw_p) - calibrate(t, base)) / (raw_p - base)
    else:
        scale = (calibrate(t, base + 1e-3) - calibrate(t, base - 1e-3)) / 2e-3
    vals = vals * scale
    cats = {}
    for c, v in zip(COLUMNS, vals):
        name = CATEGORY.get(c, "Other")
        cats[name] = cats.get(name, 0.0) + float(v)
    order = np.argsort(-np.abs(vals))[:TOP_N]
    return {"base": calibrate(t, base), "categories": cats,
            "top": [{"feature": COLUMNS[i], "value": float(X.iloc[0, i]),
                     "contribution": float(vals[i])} for i in order]}

@app.get("/api/meta")
def meta():
    return {**META, "examples": EXAMPLES, "reliability": RELIABILITY}

@app.post("/api/predict")
def predict(p: Patient):
    unknown = [k for k in p.features if k not in COLUMNS]
    if unknown:
        raise HTTPException(422, f"Unknown features: {unknown}")
    row = {**META["defaults"], **p.features}
    X = pd.DataFrame([row])[COLUMNS]
    raw = {t: float(MODELS[t].predict_proba(X)[0, 1]) for t in TARGETS}
    out = {t: calibrate(t, raw[t]) for t in TARGETS}
    out["explanations"] = {t: explain(t, X, raw[t]) for t in TARGETS}
    return out

app.mount("/", StaticFiles(directory=ROOT / "web", html=True), name="web")
