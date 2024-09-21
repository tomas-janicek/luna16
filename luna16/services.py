import datetime
import socket
from pathlib import Path

import mlflow
import svcs
from torch.utils.tensorboard.writer import SummaryWriter

from luna16 import training_logging
from luna16.settings import settings

TrainingWriter = SummaryWriter
ValidationWriter = SummaryWriter
MlFlowRun = mlflow.ActiveRun
LogMessageHandler = training_logging.LogMessageHandler


def create_registry(training_name: str) -> svcs.Registry:
    registry = svcs.Registry()

    training_start_time = datetime.datetime.now().isoformat()
    hostname = socket.gethostname()
    training_log_dir = Path(
        f"runs/{training_name}/{training_start_time}_{hostname}-training"
    )
    validation_log_dir = Path(
        f"runs/{training_name}/{training_start_time}_{hostname}-validation"
    )
    training_writer = SummaryWriter(log_dir=training_log_dir)
    validation_writer = SummaryWriter(log_dir=validation_log_dir)

    registry.register_value(TrainingWriter, training_writer)
    registry.register_value(ValidationWriter, validation_writer)

    mlflow.set_tracking_uri(uri=settings.ML_FLOW_URL)
    experiment = mlflow.set_experiment(experiment_name=training_name)
    active_run = mlflow.start_run(experiment_id=experiment.experiment_id)

    registry.register_value(MlFlowRun, active_run)

    log_message_handler = training_logging.LogMessageHandler(
        registry=registry, log_messages=training_logging.LOG_MESSAGE_HANDLERS
    )
    registry.register_value(LogMessageHandler, log_message_handler)

    return registry
