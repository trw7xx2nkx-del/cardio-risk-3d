import pandas as pd

TARGET_COLS = ["Cath", "LAD", "LCX", "RCA"]   # never used as model inputs
VHD_ORDER = {"N": 0, "mild": 1, "Moderate": 2, "Severe": 3}

def clean_features(df):
    """Raw dataset rows -> all-numeric feature table. Target columns are always removed."""
    X = df.drop(columns=[c for c in TARGET_COLS if c in df.columns]).copy()
    X = X.drop(columns=["Exertional CP"], errors="ignore")   # constant column
    X["Sex"] = X["Sex"].map({"Male": 1, "Fmale": 0, "Female": 0})
    X["VHD"] = X["VHD"].map(VHD_ORDER)
    X["BBB_LBBB"] = (X["BBB"] == "LBBB").astype(int)
    X["BBB_RBBB"] = (X["BBB"] == "RBBB").astype(int)
    X = X.drop(columns=["BBB"])
    for c in X.columns:                      # remaining text columns are Y/N
        if not pd.api.types.is_numeric_dtype(X[c]):
            X[c] = X[c].map({"Y": 1, "N": 0})
    return X

def get_targets(df):
    """1 = CAD / Stenotic, 0 = Normal."""
    return {
        "CAD": (df["Cath"] == "CAD").astype(int),
        "LAD": (df["LAD"] == "Stenotic").astype(int),
        "LCX": (df["LCX"] == "Stenotic").astype(int),
        "RCA": (df["RCA"] == "Stenotic").astype(int),
    }
