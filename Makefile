IMAGE=ml-train:local
build:
	docker build -t $(IMAGE) .
train:
	docker run --rm -v $(CURDIR)/data:/app/data -v $(CURDIR)/artifacts:/app/artifacts $(IMAGE)