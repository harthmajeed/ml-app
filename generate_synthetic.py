import numpy as np
import pandas as pd

rng = np.random.RandomState(0)
X1 = rng.normal(0, 1, 1000)
X2 = rng.normal(1, 1, 1000)

df = pd.DataFrame({"f1": X1, "f2": X2, "target": (X1+X2>0).astype(int)})
df.to_csv("data/train.csv", index=False)

pd.DataFrame({"feature": rng.normal(0, 1, 500)}).to_csv("data/ref_window.csv", index=False)
pd.DataFrame({"feature": rng.normal(0.3, 1, 500)}).to_csv("data/new_window.csv", index=False)

print("Synthetic data generated in 'data/'")
