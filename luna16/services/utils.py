import mlflow
from torch.utils.tensorboard.writer import SummaryWriter

from luna16 import training_logging
from luna16.hyperparameters_container import HyperparameterContainer

from . import creators
from .service_container import ServiceContainer

TrainingWriter = SummaryWriter
ValidationWriter = SummaryWriter
MlFlowRun = mlflow.ActiveRun
LogMessageHandler = training_logging.LogMessageHandler
Hyperparameters = HyperparameterContainer


def create_registry() -> ServiceContainer:
    registry = ServiceContainer()

    registry.register_creator(
        type=TrainingWriter,
        creator=creators.create_training_writer,
        on_registry_close=creators.clean_tensorboard_writer,
    )
    registry.register_creator(
        type=ValidationWriter,
        creator=creators.create_validation_writer,
        on_registry_close=creators.clean_tensorboard_writer,
    )
    registry.register_creator(
        type=MlFlowRun,
        creator=creators.create_mlflow_experiment,
        on_registry_close=creators.clean_mlflow_experiment,
    )

    hyperparameters = HyperparameterContainer()
    registry.register_service(Hyperparameters, hyperparameters)

    log_message_handler = training_logging.LogMessageHandler(
        registry=registry, log_messages=training_logging.LOG_MESSAGE_HANDLERS
    )
    registry.register_service(LogMessageHandler, log_message_handler)

    return registry
