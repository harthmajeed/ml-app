import pandas as pd
import joblib
from features import make_supervised

DATA_PATH = "data/passengers.csv"
MODEL_PATH = "models/ridge.joblib"

def main():
    bundle = joblib.load(MODEL_PATH)
    model = bundle["model"]
    feature_cols = bundle["feature_cols"]

    df = pd.read_csv(DATA_PATH)
    X, Y, meta, _ = make_supervised(df, max_lag=14)

    last_X = X.iloc[[-1]][feature_cols]
    last_date = meta.iloc[-1]["date"]

    pred = model.predict(last_X)[0]
    print(f"Latest known date: {last_date.date()}")
    print(f"Predicted next value (one-step): {pred:.0f}")

if __name__ == "__main__":
    main()
    