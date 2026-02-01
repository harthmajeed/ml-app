import joblib
import pandas as pd
import os

def run():
    model = joblib.load("model.joblib")
    df = pd.read_csv("data/processed.csv")
    preds = model.predict(df[["x","x2"]])
    print("predictions: ", preds[:3])

    # write tiny metrifs file
    metrics_dir = "metrics"
    os.makedirs(metrics_dir, exist_ok=True)
    metrics_path = os.path.join(metrics_dir, "evaluation_metrics.txt")
    with open(metrics_path, "w") as f:
        f.write(f"evaluation metrics:\n{preds}")

if __name__ == "__main__":
    run()