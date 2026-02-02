import pandas as pd
from pathlib import Path

DATA_DIR = Path("data")
OUT_DIR = Path("out")
DATA_DIR.mkdir(exist_ok=True)
OUT_DIR.mkdir(exist_ok=True)

# 1) Create a tiny raw dataset (like landing in a lake)
raw = pd.DataFrame({
    "user_id": [1,2,3,4,5,6],
    "country": ["CA","CA","US","CA","US","US"],
    "spend": [10.0, 22.5, 5.0, 13.0, 40.0, 8.0],
})
raw_path = DATA_DIR / "raw.csv"
raw.to_csv(raw_path, index=False)

# 2) Read
df = pd.read_csv(raw_path)

# 3) Transform (like a notebook cell / Spark job step)
agg = df.groupby("country")["spend"].agg(["count", "sum", "mean"])
agg.columns = ["count", "sum", "mean"]
agg = agg.reset_index()

# 4) Write curated output (like writing a Delta table)
out_path = OUT_DIR / "country_spend.csv"
agg.to_csv(out_path, index=False)

# 5) Emit “job metrics”
print("rows_in:", len(df))
print("rows_out:", len(agg))
print("output:", out_path)
print(agg)
