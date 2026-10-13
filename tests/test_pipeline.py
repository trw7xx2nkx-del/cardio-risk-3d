from pathlib import Path
import joblib
import pandas as pd
from preprocess import clean_features, get_targets

ROOT = Path(__file__).resolve().parents[1]
LEAKY = {"LAD", "LCX", "RCA", "Cath"}

def tiny_raw():
    return pd.DataFrame({
        "Age": [50, 62], "Sex": ["Male", "Fmale"], "HTN": [1, 0], "BBB": ["N", "LBBB"],
        "VHD": ["N", "Severe"], "Obesity": ["Y", "N"], "Exertional CP": ["N", "N"],
        "LAD": ["Stenotic", "Normal"], "LCX": ["Normal", "Normal"],
        "RCA": ["Normal", "Stenotic"], "Cath": ["CAD", "Normal"]})

def test_targets_never_become_features():
    X = clean_features(tiny_raw())
    assert not (LEAKY & set(X.columns))

def test_cleaning_is_numeric_and_encodes_correctly():
    X = clean_features(tiny_raw())
    assert all(pd.api.types.is_numeric_dtype(X[c]) for c in X.columns)
    assert list(X["Sex"]) == [1, 0]
    assert list(X["VHD"]) == [0, 3]
    assert list(X["BBB_LBBB"]) == [0, 1]
    assert "Exertional CP" not in X.columns

def test_target_encoding():
    y = get_targets(tiny_raw())
    assert list(y["CAD"]) == [1, 0]
    assert list(y["LAD"]) == [1, 0]
    assert list(y["RCA"]) == [0, 1]

def test_deployed_models_never_see_leaky_columns():
    cols = joblib.load(ROOT / "models" / "feature_columns.joblib")
    assert not (LEAKY & set(cols))
    assert len(cols) == len(set(cols))
