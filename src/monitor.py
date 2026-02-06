import numpy as np
import pandas as pd
import mlflow
from math import log

def psi(expected, actual, buckets=10):
    bins = np.linspace(
        min(expected.min(), actual.min()),
        max(expected.max(), actual.max()),
        buckets+1)
    
    exp_pct, _ = np.histogram(expected, bins=bins)
    act_pct, _ = np.histogram(actual, bins=bins)
    
    exp_pct = exp_pct / exp_pct.sum()
    act_pct = act_pct / act_pct.sum()

    act_pct = np.where(act_pct == 0, 1e-6, act_pct)
    exp_pct = np.where(exp_pct == 0, 1e-6, exp_pct)

    return float(np.sum((exp_pct - act_pct) * np.log(exp_pct / act_pct)))

if __name__ == "__main__":
    mlflow.set_experiment("monitoring")
    ref = pd.read_csv("data/ref_window.csv")["feature"].values
    new = pd.read_csv("data/new_window.csv")["feature"].values
    score = psi(pd.Series(ref), pd.Series(new), buckets=10)
    
    with mlflow.start_run():
        mlflow.log_metric("psi_feature", score)
    print("PSI logged:", score)
    