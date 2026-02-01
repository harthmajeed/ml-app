import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib
import yaml
import os

def run(params_path="params.yaml"):
    params = yaml.safe_load(open(params_path))
    df = pd.read_csv("data/processed.csv")
    X = df[["x","x2"]]
    Y = df["y"]
    model = RandomForestRegressor(n_estimators=params["train"]["n_estimators"],
                                  random_state=params["train"]["random_state"])
    model.fit(X, Y)
    joblib.dump(model, "model.joblib")

    # write tiny metrics file
    metrics_dir = "metrics"
    os.makedirs(metrics_dir, exist_ok=True)
    metrics_path = os.path.join(metrics_dir, "training_metrics.txt")
    with open(metrics_path, "w") as f:
        f.write(f"training metrics:\nn_estimators={params["train"]["n_estimators"]}\nrandom_state={params["train"]["random_state"]}")

if __name__ == "__main__":
    run()