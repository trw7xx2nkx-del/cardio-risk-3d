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
BINARY = set(META["binary"])
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

NAMES = {
    "Age": "age", "Sex": "sex", "Weight": "weight", "Length": "height", "BMI": "BMI",
    "DM": "diabetes", "HTN": "hypertension", "Current Smoker": "current smoking",
    "EX-Smoker": "past smoking", "FH": "family history of CAD", "Obesity": "obesity",
    "CRF": "chronic renal failure", "CVA": "prior stroke", "Airway disease": "airway disease",
    "Thyroid Disease": "thyroid disease", "CHF": "congestive heart failure", "DLP": "dyslipidemia",
    "BP": "blood pressure", "PR": "pulse rate", "Edema": "edema",
    "Weak Peripheral Pulse": "weak peripheral pulse", "Lung rales": "lung rales",
    "Systolic Murmur": "systolic murmur", "Diastolic Murmur": "diastolic murmur",
    "Typical Chest Pain": "typical chest pain", "Dyspnea": "shortness of breath",
    "Function Class": "functional class", "Atypical": "atypical chest pain",
    "Nonanginal": "non-anginal chest pain", "LowTH Ang": "low-threshold angina",
    "Q Wave": "Q wave on ECG", "St Elevation": "ST elevation", "St Depression": "ST depression",
    "Tinversion": "T-wave inversion", "LVH": "left ventricular hypertrophy",
    "Poor R Progression": "poor R-wave progression", "BBB_LBBB": "left bundle branch block",
    "BBB_RBBB": "right bundle branch block", "EF-TTE": "ejection fraction",
    "Region RWMA": "wall-motion abnormal regions", "VHD": "valvular heart disease grade",
    "FBS": "fasting blood sugar", "CR": "creatinine", "TG": "triglycerides",
    "LDL": "LDL cholesterol", "HDL": "HDL cholesterol", "BUN": "blood urea nitrogen", "ESR": "ESR",
    "HB": "hemoglobin", "K": "potassium", "Na": "sodium", "WBC": "white-cell count",
    "Lymph": "lymphocyte percentage", "Neut": "neutrophil percentage", "PLT": "platelet count",
}
SUBJECT = {"CAD": "overall coronary artery disease", "LAD": "the LAD (front of the heart)",
           "LCX": "the LCX (side and back of the heart)", "RCA": "the RCA (right side and bottom)"}
NOT_CHANGEABLE = {"Age", "Sex", "FH", "BMI", "Weight", "Length"}

app = FastAPI(title="Cardiac risk API (decision support / education only)")

class Patient(BaseModel):
    features: dict[str, float]

def calibrate(t, p):
    if t not in CAL:
        return float(p)
    q = float(np.clip(p, 1e-3, 1 - 1e-3))
    return float(1 / (1 + np.exp(-(CAL[t]["a"] * np.log(q / (1 - q)) + CAL[t]["b"]))))

def phrase(col, v):
    name = NAMES.get(col, col)
    if col in BINARY:
        if col == "Sex":
            return "male sex" if v >= 0.5 else "female sex"
        return name if v >= 0.5 else "no " + name
    return f"{name} {v:g}"

def join(items):
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]

def summarize(t, p, base, top, cats):
    ups = [phrase(r["feature"], r["value"]) for r in top if r["contribution"] >= 0.01][:3]
    downs = [phrase(r["feature"], r["value"]) for r in top if r["contribution"] <= -0.01][:2]
    s = f"Estimated probability for {SUBJECT[t]}: {p * 100:.0f}% (model baseline {base * 100:.0f}%). "
    if ups:
        s += "Pushing it up the most: " + join(ups) + ". "
    if downs:
        s += "Pulling it down: " + join(downs) + ". "
    if not ups and not downs:
        s += "No single input moved it by more than 1 point. "
    cat, v = max(cats.items(), key=lambda kv: abs(kv[1]))
    s += f"The biggest group effect came from {cat} ({v * 100:+.0f} points)."
    return s

def what_if(t, X, order, p_cal):
    rows, changes = [], []
    for i in order[:8]:
        c = COLUMNS[i]
        if c in NOT_CHANGEABLE:
            continue
        cur = float(X.iloc[0, i])
        alt = 1.0 - cur if c in BINARY else float(META["defaults"][c])
        if abs(alt - cur) < 1e-9:
            continue
        r = X.iloc[0].copy()
        r[c] = alt
        rows.append(r)
        changes.append((c, cur, alt))
    if not rows:
        return None
    probs = [calibrate(t, q) for q in MODELS[t].predict_proba(pd.DataFrame(rows))[:, 1]]
    j = int(np.argmax([abs(q - p_cal) for q in probs]))
    c, cur, alt = changes[j]
    if c in BINARY:
        change = f"{phrase(c, alt)} instead of {phrase(c, cur)}"
    else:
        change = f"{NAMES.get(c, c)} were {alt:g} (the typical value in the data) instead of {cur:g}"
    return (f"if {change}, the estimate would move from {p_cal * 100:.0f}% to {probs[j] * 100:.0f}% "
            f"({(probs[j] - p_cal) * 100:+.0f} points). This describes the model's behavior, "
            f"not medical advice or a cause-and-effect claim.")

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
    p_cal, base_cal = calibrate(t, raw_p), calibrate(t, base)
    cats = {}
    for c, v in zip(COLUMNS, vals):
        name = CATEGORY.get(c, "Other")
        cats[name] = cats.get(name, 0.0) + float(v)
    order = np.argsort(-np.abs(vals))
    top = [{"feature": COLUMNS[i], "value": float(X.iloc[0, i]),
            "contribution": float(vals[i])} for i in order[:TOP_N]]
    return {"base": base_cal, "categories": cats, "top": top,
            "text": {"summary": summarize(t, p_cal, base_cal, top, cats),
                     "whatif": what_if(t, X, order, p_cal)}}

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
