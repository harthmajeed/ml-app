import hydra
from omegaconf import DictConfig, OmegaConf
# import the Config class from schema.py
from schema import Config
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg):
    default_cfg = OmegaConf.structured(Config)
    cfg = OmegaConf.merge(default_cfg, cfg)

    print(f"-----Config used: {OmegaConf.to_yaml(cfg)}")

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
            n_estimators=cfg.rf.n_estimators,
            max_depth=cfg.rf.max_depth
        )
    elif cfg.model.name == "svm":
        model = SVC(
            kernel=cfg.svm.kernel,
            C=cfg.svm.C,
            gamma=cfg.svm.gamma
        )
    else:
        raise ValueError("Unknown model")
    print(f"-----Model Created '{cfg.model.name}'-----")

    model.fit(X_train, Y_train)
    accuracy = model.score(X_test, Y_test)
    print(f"-----Accuracy: {accuracy:.4f}-----")

if __name__ == "__main__":
    main()
