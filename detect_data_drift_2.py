import numpy as np
import joblib
from sklearn.metrics import mean_squared_error

model = joblib.load("model.joblib")

X_new = np.random.normal(50, 10, (300, 1))

Y_new = 6 * X_new.squeeze() + 5 + np.random.normal(0, 5, 300)

preds = model.predict(X_new)
mse = mean_squared_error(Y_new, preds)

print(f"MSE: {mse}")

if mse > 200:
    print("Drift Detected!")
else:
    print("no drift detected")

# Same input distribution
# Model predictions are now wrong
# Only detectable after labels arrive

# AUTOMATION FLOW
# New data →
# Run drift checks →
# If drift →
# Trigger retraining →
# Register new model →
# Redeploy

# | Your Tool | How it fits                         |
# | --------- | ----------------------------------- |
# | MLflow    | Log MSE drift metrics               |
# | Docker    | Drift checker container             |
# | Make      | `make check-drift`                  |
# | FastAPI   | `/health/drift` endpoint            |
# | DVC       | Version datasets before/after drift |
# | Hydra     | Threshold configs                   |
# | ADO       | Pipeline triggers retraining        |
