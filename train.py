import argparse, os, yaml, time

def train(cfg_path):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    print("loaded config:", cfg)
    os.makedirs("/app/artifacts", exist_ok=True)

    for epoch in range(cfg["train"]["epochs"]):
        print(f"Epoch {epoch+1}/{cfg['train']['epochs']} - lr {cfg['train']['lr']}")
        time.sleep(0.5)
    model_path = "/app/artifacts/model.txt"
    with open(model_path, "w") as f:
        f.write("dummy model content")
    print("Wrote model to", model_path)

if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args=parser.parse_args()
    train(args.config)