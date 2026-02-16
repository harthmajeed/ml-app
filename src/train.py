import pandas as pd
import joblib
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from features import make_supervised

DATA_PATH = "data/passengers.csv"
MODEL_PATH = "models/ridge.joblib"

def main():
    df = pd.read_csv(DATA_PATH)

    X, Y, meta, feature_cols = make_supervised(df, max_lag=14)

    split_idx = int(len(X) * 0.8)
    X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
    Y_train, Y_val = Y.iloc[:split_idx], Y.iloc[split_idx:]

    model = Ridge(alpha = 1.0)
    model.fit(X_train, Y_train)

    preds = model.predict(X_val)

    mae = mean_absolute_error(Y_val, preds)
    rmse = mean_squared_error(Y_val, preds)

    print(f"VAL_MAE: {mae:.2f}")
    print(f"VAL_RMSE: {rmse:.2f}")

    joblib.dump(
        {
            "model": model,
            "feature_cols": feature_cols
        },
        MODEL_PATH,
    )
    print(f"Saved model to '{MODEL_PATH}'")

if __name__ == "__main__":
    main()
    