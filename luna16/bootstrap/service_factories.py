import typing

from luna16 import message_handler, services
from luna16.hyperparameters_container import HyperparameterContainer


class ServiceFactory:
    def __init__(self, registry: services.ServiceContainer) -> None:
        self.registry = registry

    def add_tensorboard_writers(self) -> typing.Self:
        self.registry.register_creator(
            type=services.TrainingWriter,
            creator=services.create_training_writer,
            on_registry_close=services.clean_tensorboard_writer,
        )
        self.registry.register_creator(
            type=services.ValidationWriter,
            creator=services.create_validation_writer,
            on_registry_close=services.clean_tensorboard_writer,
        )
        return self

    def add_mlflow_run(self) -> typing.Self:
        self.registry.register_creator(
            type=services.MlFlowRun,
            creator=services.create_mlflow_experiment,
            on_registry_close=services.clean_mlflow_experiment,
        )
        return self

    def add_model_savers(self) -> typing.Self:
        self.registry.register_service(
            type=services.MLFlowModelSaver,
            value=services.MLFlowModelSaver(),
        )
        self.registry.register_service(
            type=services.ModelSaver,
            value=services.ModelSaver(),
        )
        return self

    def add_hyperparameters(self) -> typing.Self:
        hyperparameters = HyperparameterContainer()
        self.registry.register_service(HyperparameterContainer, hyperparameters)
        return self

    def add_message_handler(self) -> typing.Self:
        log_message_handler = message_handler.MessageHandler(
            registry=self.registry, messages=message_handler.LOG_MESSAGE_HANDLERS
        )
        self.registry.register_service(
            message_handler.MessageHandler, log_message_handler
        )
        return self
