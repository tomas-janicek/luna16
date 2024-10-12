.PHONY: format lint precommit

.DEFAULT_GOAL := lint

format:
	ruff format luna16

lint: format
	ruff check luna16 --fix

mlflow-server:
	mlflow ui --port 8080 --backend-store-uri sqlite:///mlruns.db

mlflow-migrate:
	mlflow db upgrade sqlite:///mlruns.db

tensorboard-classification:
	tensorboard --logdir=runs/Classification

tensorboard-segmentation:
	tensorboard --logdir=runs/Segmentation