# Cardio Risk 3D: coronary artery risk visualization

Multimodal AI Hackathon 2026, Track A (Cardiovascular Risk Visualization & Prediction).

A web app that predicts overall coronary artery disease (CAD) and stenosis in the three main coronary vessels (LAD, LCX, RCA) from clinical data, colors a 3D heart by the predicted risk, and explains each prediction with SHAP.

> **Clinical safety disclaimer.** Predictions are for decision support and education only. They are not a substitute for formal diagnostic imaging or a clinician's judgment. This is a hackathon prototype, not a medical device.

![App screenshot](docs/figures/app_screenshot.png)

## What it does

- **Prediction:** four classifiers (overall CAD, LAD, LCX, RCA) trained on the Extension of the Z-Alizadeh Sani dataset (303 patients).
- **3D view:** an interactive heart (Three.js) with the three coronary arteries colored from green to red by predicted stenosis probability. Rotate, zoom, and click an artery to inspect it.
- **Explanation:** for each prediction, a SHAP breakdown of the inputs that pushed the risk up or down, for overall CAD or the selected vessel.
- **Patient form:** enter patient values or load an example. The 3D view and explanation update automatically.

## Quick start

Requires Python 3.12 or newer (developed on 3.13.9) and an internet connection (Three.js loads from a CDN).

```bash
git clone https://github.com/trw7xx2nkx-del/cardio-risk-3d.git
cd cardio-risk-3d
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn api:app --port 8000
```

Open http://localhost:8000. The trained models are included in `models/`, so no training is needed to run the app. `scikit-learn` is pinned because the saved models only load reliably with the same version.

API documentation is at http://localhost:8000/docs.

## Retraining from scratch

1. Download the *Extension of Z-Alizadeh Sani Dataset* from the UCI Machine Learning Repository (also on Kaggle) and place the Excel file in `data/`.
2. `pip install jupyter openpyxl matplotlib`, then run `jupyter notebook` and run `notebooks/01_explore.ipynb` from top to bottom. It cleans the data, compares models with cross-validation, saves the models to `models/`, and writes the figures to `docs/figures/`.

## Method

**Data.** 303 patients with demographic, examination, ECG, laboratory, and echocardiography features. Targets: overall CAD (`Cath`) and stenosis of LAD, LCX, and RCA (50% or more narrowing).

**Preprocessing** (`preprocess.py`, shared by training and serving so they cannot drift apart):
- **Target leakage prevention:** `LAD`, `LCX`, `RCA`, and `Cath` are always removed from the model inputs.
- Yes/No columns become 0/1, `VHD` is ordinal (N, mild, Moderate, Severe), `BBB` becomes two indicator columns, and the constant column `Exertional CP` is dropped.
- 55 input features remain. There are no missing values in the dataset.

**Models.** Candidates (logistic regression, random forest, extra trees, histogram gradient boosting) were compared with 5-fold stratified cross-validation repeated 3 times, and the best by ROC-AUC was kept for each target. Deployed models: random forest (300 trees) for CAD, LAD, and LCX, and extra trees (500 trees) for RCA, all with balanced class weights. For LCX, histogram gradient boosting scored marginally higher (ROC-AUC 0.736 vs 0.725) but random forest was used instead because the fast SHAP tree explainer does not support it.

**Explanations.** `shap.TreeExplainer` computes per-patient contributions in probability space. The app shows the top 8 per prediction.

**Serving.** FastAPI (`api.py`) loads the models and exposes `GET /api/meta` and `POST /api/predict`. Features the user does not enter are filled with the dataset median, so predictions depend on those defaults.

## Results

Out-of-fold predictions from stratified 5-fold cross-validation (every patient is scored by a model that never saw them). Threshold 0.5.

| Target | Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Brier |
|---|---|---|---|---|---|---|---|
| Overall CAD | Random forest | 0.851 | 0.887 | 0.907 | 0.897 | 0.924 | 0.113 |
| LAD | Random forest | 0.802 | 0.810 | 0.864 | 0.836 | 0.856 | 0.163 |
| LCX | Random forest | 0.686 | 0.605 | 0.580 | 0.592 | 0.718 | 0.212 |
| RCA | Extra trees | 0.670 | 0.600 | 0.368 | 0.457 | 0.725 | 0.201 |

Figures are in `docs/figures/`: `roc.png`, `confusion.png`, `calibration.png`, and `shap_CAD.png`, `shap_LAD.png`, `shap_LCX.png`, `shap_RCA.png`. The full metrics table is `docs/final_metrics.csv`.

## Limitations

- **Small, single-source dataset.** 303 patients, no external validation. Results may not generalize to other hospitals or populations.
- **LCX and RCA are weak.** Their ROC-AUC (about 0.72) means the model is only modestly better than chance at telling which specific vessel is narrowed, and RCA recall at the 0.5 threshold is only 0.37. Treat these two colors as weak signals.
- **Calibration is imperfect.** The probabilities are less extreme than reality: at the high end, observed disease rates are higher than the predicted values, and at the low end they are lower (see `calibration.png`).
- **Baseline is not the real prevalence.** Class balancing makes the model's baseline 50%, whereas 71% of patients in the dataset have CAD.
- **Model selection is slightly optimistic.** The best model was picked using the same cross-validation data used to report results.
- **SHAP shows associations, not causes,** and the lab-value drivers for LCX and RCA are small and may be noise at this sample size.
- **Stylized anatomy.** The heart is a schematic shape and the arteries are drawn paths. Colors show model output per vessel, not the location of a lesion; the dataset has no spatial lesion labels.
- **Median defaults.** Inputs the user does not change are filled with the dataset median.
- **Example patients** in the dropdown come from the training data and are for demonstration only.

## Repository layout

```
api.py                FastAPI server (predictions + SHAP)
preprocess.py         Shared data cleaning
requirements.txt
models/               Trained models and feature metadata
notebooks/01_explore.ipynb   Exploration, training, evaluation
web/index.html        3D viewer and dashboard (Three.js)
docs/                 Figures and metrics
```

## Data, licenses, and credits

- Dataset: Extension of the Z-Alizadeh Sani Dataset, UCI Machine Learning Repository. TODO: paste the dataset's citation and license text from its UCI page here.
- 3D rendering: [Three.js](https://threejs.org) (MIT), loaded from a CDN. The heart geometry is generated in code; no third-party 3D model is used.
- Libraries: scikit-learn, SHAP, FastAPI, pandas.
