import os, hydra, mlflow, joblib, pandas
from dataclasses import dataclass
from omegaconf import DictConfig
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

@dataclass
class Paths:
    data_path: str = "data/train.csv"
    model_out: str = "models/model.joblib"

@hydra.main(config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    paths = Paths(**cfg.paths)
    mlflow.set_experiment(cfg.mlflow.experiment_name)
    with mlflow.start_run():
        df = pd.read_csv(paths.data_path)
        X = df.drop(cfg.target_col, axis=1)
        Y = df[cfg.target_col]
        X_train, X_val, Y_train, Y_val = train_test_split(
            X, Y,
            test_size = cfg.train.test_size,
            random_state = cfg.train.seed)

        model = RandomForestClassifier(
            n_estimators = cfg.model.n_estimators,
            random_state = cfg.train.seed)
        model.fit(X_train, Y_train)
        preds = model.predict(X_val)
        acc = accuracy_score(Y_val, preds)
        mlflow.log_metric("val_accuracy", float(acc))
        
        os.makedirs(os.path.dirname(paths.model_out), exist_ok=True)
        joblib.dump(model, paths.model_out)
        mlflow.log_artifact(paths.model_out, artifact_path="model")
        print("Training done. Val accuracy: ", acc)

if __name__ == "__main__":
    main()
    
