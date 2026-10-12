"""Train all models with nested cross-validation; save models, metrics, figures.
Run from the project root:  python train.py"""
import json
from pathlib import Path
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.calibration import calibration_curve
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, brier_score_loss,
                             confusion_matrix, precision_recall_fscore_support,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from preprocess import clean_features, get_targets

ROOT, SEED = Path(__file__).parent, 42
TARGETS = ["CAD", "LAD", "LCX", "RCA"]
(ROOT / "models").mkdir(exist_ok=True)
(ROOT / "docs" / "figures").mkdir(parents=True, exist_ok=True)

data_file = next(p for p in sorted((ROOT / "data").iterdir())
                 if p.suffix.lower() in {".xlsx", ".xls", ".csv"})
df = pd.read_csv(data_file) if data_file.suffix.lower() == ".csv" else pd.read_excel(data_file)
X, y = clean_features(df), get_targets(df)
print("Data:", data_file.name, X.shape)

GRID = {"min_samples_leaf": [1, 3, 5], "max_features": ["sqrt", 0.3],
        "class_weight": [None, "balanced"]}
CANDIDATES = {
    "RandomForest": RandomForestClassifier(n_estimators=300, random_state=SEED),
    "ExtraTrees": ExtraTreesClassifier(n_estimators=400, random_state=SEED),
}
outer = StratifiedKFold(5, shuffle=True, random_state=SEED)
inner = StratifiedKFold(3, shuffle=True, random_state=SEED)

def search(est, cv):
    return GridSearchCV(est, GRID, cv=cv, scoring="roc_auc", n_jobs=-1)

def ci(yt, p, fn, n=1000):
    rng, vals = np.random.default_rng(SEED), []
    for _ in range(n):
        s = rng.integers(0, len(yt), len(yt))
        if len(np.unique(yt[s])) == 2:
            vals.append(fn(yt[s], p[s]))
    return np.percentile(vals, [2.5, 97.5])

final, oof, rows, info = {}, {}, [], {}
for t in TARGETS:
    yt = y[t].values
    best_auc = -1
    for name, est in CANDIDATES.items():
        p = cross_val_predict(search(est, inner), X, yt, cv=outer, method="predict_proba")[:, 1]
        auc = roc_auc_score(yt, p)
        print(f"{t:4s} {name:12s} nested-CV AUC {auc:.3f}")
        if auc > best_auc:
            best_auc, best_name, best_p = auc, name, p
    gs = search(CANDIDATES[best_name], StratifiedKFold(5, shuffle=True, random_state=SEED)).fit(X, yt)
    final[t], oof[t] = gs.best_estimator_, best_p
    joblib.dump(gs.best_estimator_, ROOT / "models" / f"{t}.joblib")
    pred = best_p >= 0.5
    pr, rc, f1, _ = precision_recall_fscore_support(yt, pred, average="binary")
    lo, hi = ci(yt, best_p, roc_auc_score)
    rows.append({"target": t, "model": best_name, "accuracy": accuracy_score(yt, pred),
                 "precision": pr, "recall": rc, "f1": f1, "roc_auc": auc if False else best_auc,
                 "auc_ci_low": lo, "auc_ci_high": hi, "brier": brier_score_loss(yt, best_p)})
    info[t] = {"model": best_name, "params": {k: (v if v is None or isinstance(v, (int, float, str)) else str(v))
                                             for k, v in gs.best_params_.items()}}
    print(f"  -> {t}: {best_name} {gs.best_params_}")

metrics = pd.DataFrame(rows).round(3)
metrics.to_csv(ROOT / "docs" / "final_metrics.csv", index=False)
json.dump(info, open(ROOT / "models" / "model_info.json", "w"), indent=1)
print(metrics.to_string())

# Serving metadata (same format as before)
joblib.dump(list(X.columns), ROOT / "models" / "feature_columns.joblib")
meta = {"columns": list(X.columns),
        "defaults": {c: float(X[c].median()) for c in X.columns},
        "ranges": {c: [float(X[c].min()), float(X[c].max())] for c in X.columns},
        "binary": [c for c in X.columns if set(X[c].unique()) <= {0, 1}]}
json.dump(meta, open(ROOT / "models" / "feature_meta.json", "w"), indent=1)
vessels = y["LAD"] + y["LCX"] + y["RCA"]
groups = {"Multi-vessel disease (from dataset)": vessels == 3,
          "LAD only (from dataset)": (y["LAD"] == 1) & (y["LCX"] == 0) & (y["RCA"] == 0),
          "No disease (from dataset)": (y["CAD"] == 0) & (vessels == 0)}
examples = {n: {k: float(v) for k, v in X[m].iloc[0].items()} for n, m in groups.items() if m.any()}
json.dump(examples, open(ROOT / "models" / "examples.json", "w"), indent=1)

# Figures
fig, ax = plt.subplots(figsize=(6, 5))
for t in TARGETS:
    fpr, tpr, _ = roc_curve(y[t], oof[t])
    ax.plot(fpr, tpr, label=f"{t} (AUC {roc_auc_score(y[t], oof[t]):.2f})")
ax.plot([0, 1], [0, 1], "k--", lw=1)
ax.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC (nested 5-fold CV)")
ax.legend(loc="lower right"); fig.tight_layout(); fig.savefig(ROOT / "docs/figures/roc.png", dpi=200); plt.close(fig)

fig, axes = plt.subplots(2, 2, figsize=(8, 7))
for ax, t in zip(axes.ravel(), TARGETS):
    ConfusionMatrixDisplay(confusion_matrix(y[t], oof[t] >= 0.5), display_labels=["No", "Yes"]).plot(ax=ax, colorbar=False)
    ax.set_title(t)
fig.tight_layout(); fig.savefig(ROOT / "docs/figures/confusion.png", dpi=200); plt.close(fig)

fig, ax = plt.subplots(figsize=(6, 5))
for t in TARGETS:
    frac, mean = calibration_curve(y[t], oof[t], n_bins=6, strategy="quantile")
    ax.plot(mean, frac, marker="o", label=f"{t} (Brier {brier_score_loss(y[t], oof[t]):.2f})")
ax.plot([0, 1], [0, 1], "k--", lw=1)
ax.set(xlabel="Predicted probability", ylabel="Observed fraction with disease", title="Calibration (nested 5-fold CV)")
ax.legend(); fig.tight_layout(); fig.savefig(ROOT / "docs/figures/calibration.png", dpi=200); plt.close(fig)

for t in TARGETS:
    ex = shap.TreeExplainer(final[t])(X, check_additivity=False)
    v = ex.values[:, :, 1] if ex.values.ndim == 3 else ex.values
    plt.figure(); shap.summary_plot(v, X, show=False, max_display=12)
    plt.title(f"{t}: global feature impact (SHAP)")
    plt.savefig(ROOT / f"docs/figures/shap_{t}.png", dpi=200, bbox_inches="tight"); plt.close()
print("Done.")
