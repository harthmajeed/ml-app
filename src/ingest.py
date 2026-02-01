import pandas as pd

def run():
    # Create small synthetic data
    df = pd.DataFrame({"x":[1,2,3,4,5,6], "y":[2,4,6,8,10,12]})
    # save to raw.csv
    df.to_csv("data/raw.csv", index=False)
    
if __name__ == "__main__":
    run()