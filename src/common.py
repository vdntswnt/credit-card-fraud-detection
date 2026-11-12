"""Shared helpers for the fraud detection project."""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "creditcard.csv"
MODEL_DIR = ROOT / "models"
FIG_DIR = ROOT / "reports" / "figures"

PCA_COLS = [f"V{i}" for i in range(1, 29)]


def load_data(path: Path = DATA_PATH, drop_duplicates: bool = True) -> pd.DataFrame:
    """Load the Kaggle ULB credit card dataset, sorted by time."""
    df = pd.read_csv(path)
    if drop_duplicates:
        df = df.drop_duplicates()
    return df.sort_values("Time").reset_index(drop=True)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add engineered features. Works on a frame with Time and Amount."""
    out = df.copy()
    out["log_amount"] = np.log1p(out["Amount"])
    out["hour"] = (out["Time"] // 3600) % 24
    return out


# NOTE: binary/discrete engineered flags (e.g. "is_round_amount") were tried and removed -
# SMOTE interpolates between rows, so synthetic frauds never have round amounts and the
# model learned a spurious "round amount => legit" rule (visible in the SHAP plot).
FEATURES = PCA_COLS + ["log_amount", "hour"]


def time_split(df: pd.DataFrame, train_frac: float = 0.6, val_frac: float = 0.2):
    """Chronological split so we never train on the future."""
    n = len(df)
    i1, i2 = int(n * train_frac), int(n * (train_frac + val_frac))
    return df.iloc[:i1], df.iloc[i1:i2], df.iloc[i2:]
