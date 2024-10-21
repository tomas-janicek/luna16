import datetime
import socket
import typing
from pathlib import Path

import mlflow
from torch.utils.tensorboard.writer import SummaryWriter

from luna16 import settings, utils


def create_training_writer(
    training_name: str, training_start_time: datetime.datetime, **kwargs: typing.Any
) -> SummaryWriter:
    hostname = socket.gethostname()
    training_log_dir = Path(
        f"runs/{training_name}/{utils.get_datetime_string(training_start_time)}_{hostname}-training"
    )
    training_writer = SummaryWriter(log_dir=training_log_dir)
    return training_writer


def create_validation_writer(
    training_name: str, training_start_time: datetime.datetime, **kwargs: typing.Any
) -> SummaryWriter:
    hostname = socket.gethostname()
    validation_log_dir = Path(
        f"runs/{training_name}/{utils.get_datetime_string(training_start_time)}_{hostname}-validation"
    )
    validation_writer = SummaryWriter(log_dir=validation_log_dir)
    return validation_writer


def clean_tensorboard_writer(writer: SummaryWriter) -> None:
    writer.close()


def create_mlflow_experiment(
    training_name: str, training_start_time: datetime.datetime, **kwargs: typing.Any
) -> mlflow.ActiveRun:
    mlflow.set_tracking_uri(uri=settings.ML_FLOW_URL)
    experiment = mlflow.set_experiment(experiment_name=training_name)
    active_run = mlflow.start_run(
        experiment_id=experiment.experiment_id,
        run_name=f"{training_name}-{utils.get_datetime_string(training_start_time)}",
        tags={"version": "0.0.1"},
        log_system_metrics=settings.MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING,
    )
    return active_run


def clean_mlflow_experiment(active_run: mlflow.ActiveRun) -> None:
    mlflow.end_run()
