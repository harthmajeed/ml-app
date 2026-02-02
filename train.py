import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import joblib

np.random.seed(42)

X_train = np.random.normal(loc=50, scale=10, size=(1000,1))
Y_train = 3 * X_train.squeeze() + 10 + np.random.normal(0, 5, 1000)

model = LinearRegression()
model.fit(X_train, Y_train)

joblib.dump(model, "model.joblib")
print("Model trained and saved")