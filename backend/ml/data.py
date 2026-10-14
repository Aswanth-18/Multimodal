import pandas as pd
from pathlib import Path
from .features import F, TARGETS, POSITIVE, LEAKAGE_COLUMNS

DATA = Path(__file__).resolve().parents[1] / "data" / "z_alizadeh_sani_extension.xlsx"

def load():
    """Return (X, Y) where X contains only legitimate input features (no leakage columns)."""
    d = pd.read_excel(DATA, sheet_name=0)
    d.columns = [c.strip() for c in d.columns]
    Y = pd.DataFrame({k: (d[col] == POSITIVE[col]).astype(int) for k, col in TARGETS.items()})
    X = d.drop(columns=LEAKAGE_COLUMNS).copy()
    X["Sex"] = (X["Sex"] == "Male").astype(int)
    X["LBBB"] = (X["BBB"] == "LBBB").astype(int)
    X["RBBB"] = (X["BBB"] == "RBBB").astype(int)
    X["VHD"] = X["VHD"].str.lower().map({"n": 0, "mild": 1, "moderate": 2, "severe": 3})
    X = X.drop(columns=["BBB"])
    for c in X.columns:
        if not pd.api.types.is_numeric_dtype(X[c]):
            X[c] = X[c].map({"Y": 1, "N": 0})
    X = X.loc[:, X.nunique() > 1]          # drop constant columns (e.g. Exertional CP)
    X = X[[c for c in F if c in X.columns]]  # schema order
    assert not set(X.columns) & set(LEAKAGE_COLUMNS)
    assert X.isna().sum().sum() == 0, X.isna().sum()[lambda s: s > 0]
    return X, Y
