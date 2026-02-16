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