# ONE FOR ALL - SCRIPT THAT WILL CREATE FILES AND SCRIPTS FOR THE ENTIRE PROJECT

#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="ml-docker-demo"
IMAGE_NAME="ml-train:local"

# create project folder
rm -rf "$PROJECT_DIR"
mkdir -p "$PROJECT_DIR"
cd "$PROJECT_DIR"

# pyproject.toml
cat > pyproject.toml <<'PYPROJECT'
[build-system]
requires = ["setuptools", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "ml-docker-demo"
version = "0.1"
description = "Simple ML training on Docker demo"
dependencies = ["pyyaml"]
PYPROJECT

# requirements.txt
cat > requirements.txt <<'REQ'
# minimal runtime deps for demo
pyyaml
REQ

# config.yaml
cat > config.yaml <<'YAML'
train:
  epochs: 1
  lr: 0.01
YAML

# train.py
cat > train.py <<'PY'
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
PY

# entrypoint.sh
cat > entrypoint.sh <<'SH'
#!/usr/bin/env bash
set -e
if [ $# -eq 0 ]; then
  python train.py --config config.yaml
else
  exec "$@"
fi
SH
chmod +x entrypoint.sh

# Dockerfile
cat > Dockerfile <<'DOCK'
FROM python:3.10-slim
WORKDIR /app
COPY pyproject.toml ./
RUN pip install --no-cache-dir .
COPY . /app
RUN chmod +x /app/entrypoint.sh
ENTRYPOINT ["/app/entrypoint.sh"]
DOCK

# Makefile (optional)
cat > Makefile <<'MK'
IMAGE=ml-train:local
build:
	docker build -t $(IMAGE) .
train:
	docker run --rm -v $(CURDIR)/data:/app/data -v $(CURDIR)/artifacts:/app/artifacts $(IMAGE)
MK

# create host mount dirs
mkdir -p artifacts data

# build and run
echo "Building Docker image: ${IMAGE_NAME}"
make build

echo "Running container (artifacts will appear in ./artifacts)"
make train

echo "Done. Inspect ./artifacts/model.txt"
ls -la artifacts || true