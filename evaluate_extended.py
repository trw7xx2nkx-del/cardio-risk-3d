"""Extended evaluation: baselines, subgroup performance, screening-style thresholds.
Run from the project root after train.py:  python evaluate_extended.py"""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from preprocess import clean_features, get_targets

ROOT = Path(__file__).parent
TARGETS = ["CAD", "LAD", "LCX", "RCA"]
CLS = {"RandomForest": (RandomForestClassifier, 300), "ExtraTrees": (ExtraTreesClassifier, 400)}
SEEDS = range(3)
(ROOT / "docs" / "figures").mkdir(parents=True, exist_ok=True)

data_file = next(p for p in sorted((ROOT / "data").iterdir())
                 if p.suffix.lower() in {".xlsx", ".xls", ".csv"})
df = pd.read_csv(data_file) if data_file.suffix.lower() == ".csv" else pd.read_excel(data_file)
X, y = clean_features(df), get_targets(df)
info = json.load(open(ROOT / "models" / "model_info.json"))
final = pd.read_csv(ROOT / "docs" / "final_metrics.csv").set_index("target")

def folds(seed):
    return StratifiedKFold(5, shuffle=True, random_state=seed)

# Out-of-fold predictions of the deployed models (3 different fold splits)
oof = {}
for t in TARGETS:
    cls, n = CLS[info[t]["model"]]
    oof[t] = [cross_val_predict(cls(n_estimators=n, random_state=42, **info[t]["params"]), X, y[t].values,
                                cv=folds(s), method="predict_proba", n_jobs=-1)[:, 1] for s in SEEDS]
print("Model predictions done.")

def boot_auc(yt, p, n=1000):
    rng, vals = np.random.default_rng(42), []
    for _ in range(n):
        s = rng.integers(0, len(yt), len(yt))
        if len(np.unique(yt[s])) == 2:
            vals.append(roc_auc_score(yt[s], p[s]))
    return np.percentile(vals, [2.5, 97.5])

# 1) Baselines
SIMPLE = ["Age", "Sex", "HTN", "DM", "Current Smoker", "FH", "Typical Chest Pain"]
def cv_auc(cols, yt):
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    return float(np.mean([roc_auc_score(yt, cross_val_predict(model, X[cols], yt, cv=folds(s),
                          method="predict_proba")[:, 1]) for s in SEEDS]))
rows = []
for t in TARGETS:
    yt = y[t].values
    rows.append({"target": t,
                 "typical chest pain alone": round(roc_auc_score(yt, X["Typical Chest Pain"]), 3),
                 "age alone (logistic)": round(cv_auc(["Age"], yt), 3),
                 "7 simple risk factors (logistic)": round(cv_auc(SIMPLE, yt), 3),
                 "our model (nested CV)": float(final.loc[t, "roc_auc"])})
baselines = pd.DataFrame(rows)
baselines.to_csv(ROOT / "docs" / "baselines.csv", index=False)
print("\nROC-AUC against simple baselines:\n", baselines.to_string(index=False))

# 2) Subgroups (each patient's score = average of their out-of-fold predictions)
mean_oof = {t: np.mean(oof[t], axis=0) for t in TARGETS}
groups = {"Female": X["Sex"] == 0, "Male": X["Sex"] == 1, "Age under 50": X["Age"] < 50,
          "Age 50-59": X["Age"].between(50, 59), "Age 60-69": X["Age"].between(60, 69),
          "Age 70+": X["Age"] >= 70}
rows = []
for t in TARGETS:
    for g, mask in groups.items():
        m = mask.values
        yt, p = y[t].values[m], mean_oof[t][m]
        pos = int(yt.sum())
        ok = min(pos, len(yt) - pos) >= 8
        lo, hi = boot_auc(yt, p) if ok else (np.nan, np.nan)
        rows.append({"target": t, "group": g, "n": int(m.sum()), "prevalence": round(float(yt.mean()), 2),
                     "auc": round(roc_auc_score(yt, p), 3) if ok else np.nan,
                     "ci_low": round(lo, 3), "ci_high": round(hi, 3)})
sub = pd.DataFrame(rows)
sub.to_csv(ROOT / "docs" / "subgroups.csv", index=False)
print("\nPerformance by subgroup (blank = too few cases to estimate):\n", sub.to_string(index=False))

fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
for ax, t in zip(axes.ravel(), TARGETS):
    d = sub[(sub.target == t) & sub.auc.notna()].reset_index(drop=True)
    ax.errorbar(d.auc, range(len(d)), xerr=[d.auc - d.ci_low, d.ci_high - d.auc], fmt="o", capsize=3)
    ax.set_yticks(range(len(d)))
    ax.set_yticklabels([f"{g} (n={n})" for g, n in zip(d.group, d.n)])
    ax.axvline(0.5, color="k", ls="--", lw=1)
    ax.set_title(t); ax.set_xlabel("ROC-AUC (95% bootstrap CI)")
fig.tight_layout(); fig.savefig(ROOT / "docs" / "figures" / "subgroup_auc.png", dpi=200)

# 3) Screening-style operating points (threshold chosen on training folds, tested on held-out folds)
def pick_threshold(p, yt, sens):
    pos = np.sort(p[yt == 1])
    return pos[int(np.floor((1 - sens) * len(pos)))]
rows = []
for t in TARGETS:
    yt = y[t].values
    for sens in (0.90, 0.80):
        tp = fp = tn = fn = 0
        for s, p in enumerate(oof[t]):
            for tr, te in StratifiedKFold(5, shuffle=True, random_state=50 + s).split(p, yt):
                pred = p[te] >= pick_threshold(p[tr], yt[tr], sens)
                tp += int((pred & (yt[te] == 1)).sum()); fp += int((pred & (yt[te] == 0)).sum())
                tn += int((~pred & (yt[te] == 0)).sum()); fn += int((~pred & (yt[te] == 1)).sum())
        rows.append({"target": t, "aimed sensitivity": sens,
                     "sensitivity": round(tp / (tp + fn), 3), "specificity": round(tn / (tn + fp), 3),
                     "PPV": round(tp / (tp + fp), 3), "NPV": round(tn / (tn + fn), 3)})
screen = pd.DataFrame(rows)
screen.to_csv(ROOT / "docs" / "screening_thresholds.csv", index=False)
print("\nScreening-style operating points (held-out):\n", screen.to_string(index=False))
print("\nSaved docs/baselines.csv, subgroups.csv, screening_thresholds.csv and figures/subgroup_auc.png")
