"""Fit Platt calibration per target on cross-validated predictions.
Run from the project root after train.py:  python calibrate.py"""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from preprocess import clean_features, get_targets

ROOT = Path(__file__).parent
TARGETS = ["CAD", "LAD", "LCX", "RCA"]
CLS = {"RandomForest": (RandomForestClassifier, 300), "ExtraTrees": (ExtraTreesClassifier, 400)}

data_file = next(p for p in sorted((ROOT / "data").iterdir())
                 if p.suffix.lower() in {".xlsx", ".xls", ".csv"})
df = pd.read_csv(data_file) if data_file.suffix.lower() == ".csv" else pd.read_excel(data_file)
X, y = clean_features(df), get_targets(df)
info = json.load(open(ROOT / "models" / "model_info.json"))

def logit(p):
    p = np.clip(p, 1e-3, 1 - 1e-3)
    return np.log(p / (1 - p))

def platt(p, yt):
    lr = LogisticRegression(C=1e6).fit(logit(p).reshape(-1, 1), yt)
    return float(lr.coef_[0, 0]), float(lr.intercept_[0])

def apply(p, a, b):
    return 1 / (1 + np.exp(-(a * logit(p) + b)))

def cv_calibrate(p, yt, seed):
    out = np.zeros_like(p)
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=100 + seed).split(p, yt):
        a, b = platt(p[tr], yt[tr])
        out[te] = apply(p[te], a, b)
    return out

result, plot_data = {}, {}
for t in TARGETS:
    yt = y[t].values
    cls, n = CLS[info[t]["model"]]
    oofs = [cross_val_predict(cls(n_estimators=n, random_state=42, **info[t]["params"]), X, yt,
                              cv=StratifiedKFold(5, shuffle=True, random_state=s),
                              method="predict_proba", n_jobs=-1)[:, 1] for s in range(3)]
    a, b = platt(np.concatenate(oofs), np.tile(yt, 3))
    cals = [cv_calibrate(o, yt, s) for s, o in enumerate(oofs)]
    raw = float(np.mean([brier_score_loss(yt, o) for o in oofs]))
    cal = float(np.mean([brier_score_loss(yt, c) for c in cals]))
    result[t] = {"a": a, "b": b, "brier_raw": raw, "brier_cal": cal}
    plot_data[t] = (yt, oofs[0], cals[0])
    print(f"{t}: Brier raw {raw:.3f} -> calibrated {cal:.3f}   (a={a:.2f}, b={b:.2f})")

json.dump(result, open(ROOT / "models" / "calibration.json", "w"), indent=1)

fig, axes = plt.subplots(2, 2, figsize=(9, 8))
for ax, t in zip(axes.ravel(), TARGETS):
    yt, raw, cal = plot_data[t]
    for p, lab in [(raw, "Raw model"), (cal, "Calibrated")]:
        frac, mean = calibration_curve(yt, p, n_bins=6, strategy="quantile")
        ax.plot(mean, frac, marker="o", label=lab)
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set(title=t, xlabel="Predicted probability", ylabel="Observed fraction")
    ax.legend()
fig.tight_layout()
(ROOT / "docs" / "figures").mkdir(parents=True, exist_ok=True)
fig.savefig(ROOT / "docs" / "figures" / "calibration_before_after.png", dpi=200)
print("Saved models/calibration.json and docs/figures/calibration_before_after.png")
