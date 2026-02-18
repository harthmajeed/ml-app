import pandas as pd

def make_supervised(df: pd.DataFrame, 
                    target_col: str = "total",
                    max_lag: int = 14):
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

    # basic sanity
    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
    df = df.dropna(subset=[target_col])

    # lag features
    for lag in range(1, max_lag + 1):
        df[f"lag_{lag}"] = df[target_col].shift(lag)

    # rolling stats
    df["roll_mean_7"] = df[target_col].shift(1).rolling(7).mean()
    df["roll_std_7"] = df[target_col].shift(1).rolling(7).std()

    df = df.dropna().reset_index(drop=True)

    feature_cols = [c for c in df.columns if c.startswith("lag_")] + ["roll_mean_7", "roll_std_7"]

    X = df[feature_cols]
    Y = df[target_col]
    meta = df[["date"]]
    return X, Y, meta, feature_cols

import numpy as np

def make_feature_row_from_history(history, max_lag: int = 14):
    """
    history: list/array of past target values in chronological order (old -> new)
    returns: dict of feature_name -> value for the next prediction step
    """
    if len(history) < max_lag + 7:
        raise ValueError(f"Need at least {max_lag + 7} history points, got {len(history)}")

    h = np.array(history, dtype=float)

    feats = {}
    # lag_1 = last value, lag_2 = second last, ...
    for lag in range(1, max_lag + 1):
        feats[f"lag_{lag}"] = h[-lag]

    last_7 = h[-7:]
    feats["roll_mean_7"] = float(last_7.mean())
    feats["roll_std_7"] = float(last_7.std(ddof=1)) if len(last_7) > 1 else 0.0
    return feats
