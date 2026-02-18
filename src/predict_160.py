import pandas as pd
import joblib
from pathlib import Path
from features import make_feature_row_from_history

DATA_PATH = "data/passengers.csv"
MODEL_PATH = "models/ridge.joblib"
OUT_PATH = "outputs/forecast_160d.csv"

HORIZON_DAYS = 160
MAX_LAG = 14

def main():
    Path("outputs").mkdir(exist_ok=True)

    bundle = joblib.load(MODEL_PATH)
    model = bundle["model"]
    feature_cols = bundle["feature_cols"]

    df = pd.read_csv(DATA_PATH)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

    # Use only the target series as history
    history = df["total"].astype(float).tolist()

    last_date = df["date"].iloc[-1]

    forecasts = []
    for step in range(1, HORIZON_DAYS + 1):
        feats = make_feature_row_from_history(history, max_lag=MAX_LAG)

        # Ensure column order matches training
        X_row = pd.DataFrame([feats])[feature_cols]

        yhat = float(model.predict(X_row)[0])

        forecast_date = last_date + pd.Timedelta(days=step)
        forecasts.append({"date": forecast_date.date().isoformat(), "total": yhat})

        print(f"Step {step}")
        print("Lag_1:", feats["lag_1"])
        print("Roll mean:", feats["roll_mean_7"])
        print("Forecast date:", forecast_date)
        print("Prediction:", yhat)
        print("------")

        # Recursive rollout: append prediction into history
        history.append(yhat)

    out_df = pd.DataFrame(forecasts)
    out_df.to_csv(OUT_PATH, index=False)

    print(f"Saved {len(out_df)} day forecast to {OUT_PATH}")
    print(out_df.head(5))
    print("...")
    print(out_df.tail(5))

if __name__ == "__main__":
    main()
