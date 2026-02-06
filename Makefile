.PHONY: data train build serve monitor mlflow

data:
	python generate_synthetic.py

train:
	python src/train.py

build:
	docker build -t mlops-demo:latest .

serve:
	docker run -p 8000:8000 mlops-demo:latest

monitor:
	python src/monitor.py

mlflow:
	mlflow ui --port 5000
	