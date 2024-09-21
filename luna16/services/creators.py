import datetime
import socket
import typing
from pathlib import Path

import mlflow
from torch.utils.tensorboard.writer import SummaryWriter

from luna16.settings import settings


def create_training_writer(
    training_name: str, training_start_time: datetime.datetime, **kwargs: typing.Any
) -> SummaryWriter:
    hostname = socket.gethostname()
    training_log_dir = Path(
        f"runs/{training_name}/{training_start_time}_{hostname}-training"
    )
    training_writer = SummaryWriter(log_dir=training_log_dir)
    return training_writer


def create_validation_writer(
    training_name: str, training_start_time: datetime.datetime, **kwargs: typing.Any
) -> SummaryWriter:
    hostname = socket.gethostname()
    validation_log_dir = Path(
        f"runs/{training_name}/{training_start_time}_{hostname}-validation"
    )
    validation_writer = SummaryWriter(log_dir=validation_log_dir)
    return validation_writer


def create_mlflow_experiment(training_name: str) -> mlflow.ActiveRun:
    mlflow.set_tracking_uri(uri=settings.ML_FLOW_URL)
    experiment = mlflow.set_experiment(experiment_name=training_name)
    active_run = mlflow.start_run(experiment_id=experiment.experiment_id)
    return active_run
