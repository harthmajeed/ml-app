import pandas as pd

def run():
    df = pd.read_csv("data/raw.csv")
    df["x2"] = df["x"] ** 2
    df.to_csv("data/processed.csv", index=False)

if __name__ == "__main__":
    run()