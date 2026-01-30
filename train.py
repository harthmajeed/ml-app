# import mlflow
# import mlflow.sklearn
import hydra
from omegaconf import DictConfig
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig):
    print(f"-----Config used: {cfg}")

    data = load_iris(as_frame=True)
    X = data.data
    Y = data.target
    print("-----Loaded IRIS dataset-----")

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, 
        test_size=cfg.dataset.test_size, 
        random_state=cfg.dataset.random_state)

    if cfg.model.name == "random_forest":
        model = RandomForestClassifier(
            n_estimators=cfg.model.n_estimators,
            max_depth=cfg.model.max_depth
        )
    elif cfg.model.name == "svm":
        model = SVC(
            kernel=cfg.model.kernel,
            C=cfg.model.C,
            gamma=cfg.model.gamma
        )
    else:
        raise ValueError("Unknown model")
    print(f"-----Model Created '{cfg.model.name}'-----")

    model.fit(X_train, Y_train)
    accuracy = model.score(X_test, Y_test)
    print(f"-----Accuracy: {accuracy:.4f}-----")

if __name__ == "__main__":
    main()


# def train(n_estimators, max_depth=None):
#      load_data()
#     classifier = RandomForestClassifier(n_estimators=n_estimators,
#                                       max_depth=max_depth,
#                                       random_state=42)
#     print("Training Random Forest Classifier")
#     classifier.fit(X_train, Y_train)
#     print("Running Predictions on Classifier")
#     predictions = classifier.predict(X_test)
#     accuracy = accuracy_score(Y_test, predictions)
#     return classifier, accuracy

# if __name__ == "__main__":
#     # Read env/cli overrides
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--n_estimators", type=int, default=50)
#     parser.add_argument("--max_depth", type=int, default=None)
#     parser.add_argument("--run_name", type=str, default="rf_run")
#     args = parser.parse_args()

#     # MLFlow tracking URI is by default set to "./mlruns" if we don't specify it, here we are being explicit
#     mlflow.set_tracking_uri("file:C:\\Users\\harth\\Projects\\mlflow_store")
#     mlflow.set_experiment("iris_rf_experiment")
#     with mlflow.start_run(run_name=args.run_name) as run:
#         print("Setting Params in MLFlow")
#         mlflow.log_param("n_estimators", args.n_estimators)
#         mlflow.log_param("max_depth", args.max_depth)
#         model, accuracy = train(n_estimators=args.n_estimators, max_depth=args.max_depth)
#         mlflow.log_metric("accuracy", accuracy)

#         # Save model artifact
#         print("Saving model artifact")
#         os.makedirs("artifacts", exist_ok=True)
#         # set the model pathh
#         model_path = "artifacts/model.joblib"
#         # dump the model to the path
#         joblib.dump(model, model_path)
#         # artifact_path="model_files" - is the path within mlflow, Artifacts tab, model.joblib
#         # log_artifact is "upload this file to mlflow storage", thats it
#         # good for plots and graphs, configs, logs, data samples
#         mlflow.log_artifact(model_path, artifact_path="model_files")
#         # Log model in mlflow model registry (local)

#         print("Logging model in MLFlow")
#         # log_model is "model-aware logging", saves data about the model and enables registry, etc
#         # Log_model allows you to reload models using "load_model" function
#         mlflow.sklearn.log_model(model, "sklearn-model")
#         print(f"-----Run ID: {run.info.run_id} Accuracy: {accuracy:.4f}-----")


# mlflow ui --port 5000
# mlflow ui --port 5000 --backend-store-uri file:C:\Users\harth\Projects\mlflow_store
# http://127.0.0.1:5000
# python train.py --n_estimators 100 --max_depth 5 --run_name run_1

# # load_model_example.py
# import mlflow
# import pandas as pd
# model_uri = "runs:/f933a8991e5a4c2284b0109826fa80fe/sklearn-model"  # replace <RUN_ID> with printed run id
# model = mlflow.sklearn.load_model(model_uri)
# df = pd.DataFrame(
#     [[5.1, 3.5, 1.4, 0.2]],
#     columns=["sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)"]
# )
# print(model.predict(df))

