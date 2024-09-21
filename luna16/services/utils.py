import mlflow
from torch.utils.tensorboard.writer import SummaryWriter

from luna16 import training_logging

from . import creators
from .service_container import ServiceContainer

TrainingWriter = SummaryWriter
ValidationWriter = SummaryWriter
MlFlowRun = mlflow.ActiveRun
LogMessageHandler = training_logging.LogMessageHandler


def create_registry() -> ServiceContainer:
    registry = ServiceContainer()

    registry.register_creator(TrainingWriter, creators.create_training_writer)
    registry.register_creator(ValidationWriter, creators.create_validation_writer)
    registry.register_creator(MlFlowRun, creators.create_mlflow_experiment)

    log_message_handler = training_logging.LogMessageHandler(
        registry=registry, log_messages=training_logging.LOG_MESSAGE_HANDLERS
    )
    registry.register_service(LogMessageHandler, log_message_handler)

    return registry
