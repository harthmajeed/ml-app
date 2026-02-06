import joblib, os

def save_model(model, path = "models/model.joblib"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)

def load_model(path = "models/model.joblib"):
    return joblib.load(path)
    