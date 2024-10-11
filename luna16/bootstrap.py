import mlflow
from torch.utils.tensorboard.writer import SummaryWriter

from luna16 import message_handler, services
from luna16.hyperparameters_container import HyperparameterContainer

TrainingWriter = SummaryWriter
ValidationWriter = SummaryWriter
MlFlowRun = mlflow.ActiveRun


def create_registry() -> services.ServiceContainer:
    registry = services.ServiceContainer()

    registry.register_creator(
        type=TrainingWriter,
        creator=services.create_training_writer,
        on_registry_close=services.clean_tensorboard_writer,
    )
    registry.register_creator(
        type=ValidationWriter,
        creator=services.create_validation_writer,
        on_registry_close=services.clean_tensorboard_writer,
    )
    registry.register_creator(
        type=MlFlowRun,
        creator=services.create_mlflow_experiment,
        on_registry_close=services.clean_mlflow_experiment,
    )

    hyperparameters = HyperparameterContainer()
    registry.register_service(HyperparameterContainer, hyperparameters)

    log_message_handler = message_handler.MessageHandler(
        registry=registry, messages=message_handler.LOG_MESSAGE_HANDLERS
    )
    registry.register_service(message_handler.MessageHandler, log_message_handler)

    return registry
