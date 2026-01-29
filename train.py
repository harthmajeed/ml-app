import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuray_score
import joblib
import os

def load_data():
    data = load_iris(as_frame=True)
    X = data.data
    Y = data.target
    return train_test_split(X, Y, test_size=0.2, random_state=42)

def train(n_estimators, max_depth=None):
    X_train, X_test, Y_train, Y_test = load_data()
    