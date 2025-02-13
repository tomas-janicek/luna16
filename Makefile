.PHONY: format lint mlflow-server mlflow-migrate tensorboard-classification tensorboard-malignant-classification precommit

format:
	uv run ruff format luna16

lint: format
	uv run ruff check luna16 --fix

mlflow-server:
	uv run mlflow ui --port 8080 --backend-store-uri sqlite:///mlruns.db

mlflow-migrate:
	uv run mlflow db upgrade sqlite:///mlruns.db

tensorboard-classification:
	uv run tensorboard --logdir="runs/classification"

tensorboard-malignant-classification:
	uv run tensorboard --logdir="runs/malignant classification"