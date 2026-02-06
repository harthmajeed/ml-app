from fastapi import FastAPI
import joblib
import pandas as pd
from pydantic import BaseModel

class Input(BaseModel):
    features: List

app = FastAPI()
MODEL_PATH = "models/model.joblib"
model = joblib.load(MODEL_PATH)

@app.post("/predict")
def predict(inp: Input):
    df = pd.DataFrame([inp.features])
    preds = model.predict(df)
    return {"prediction": int(preds[0])}

@app.get("/health")
def health():
    return {"status": "ok"}
    