from luna16 import enums, message_handler, services
from luna16.hyperparameters_container import HyperparameterContainer


def create_registry(
    model_saver: enums.ModelLoader = enums.ModelLoader.FILE,
) -> services.ServiceContainer:
    registry = services.ServiceContainer()

    registry.register_creator(
        type=services.TrainingWriter,
        creator=services.create_training_writer,
        on_registry_close=services.clean_tensorboard_writer,
    )
    registry.register_creator(
        type=services.ValidationWriter,
        creator=services.create_validation_writer,
        on_registry_close=services.clean_tensorboard_writer,
    )
    registry.register_creator(
        type=services.MlFlowRun,
        creator=services.create_mlflow_experiment,
        on_registry_close=services.clean_mlflow_experiment,
    )

    # both, MLFlow and file savers are used to save new models
    registry.register_service(
        type=services.MLFlowModelSaver,
        value=services.MLFlowModelSaver(),
    )
    registry.register_service(
        type=services.ModelSaver,
        value=services.ModelSaver(),
    )

    # but only one can be used to load models when continuing training
    match model_saver:
        case enums.ModelLoader.FILE:
            registry.register_service(
                type=services.BaseModelSaver,
                value=services.ModelSaver(),
            )
        case enums.ModelLoader.ML_FLOW:
            registry.register_service(
                type=services.BaseModelSaver,
                value=services.MLFlowModelSaver(),
            )

    hyperparameters = HyperparameterContainer()
    registry.register_service(HyperparameterContainer, hyperparameters)

    log_message_handler = message_handler.MessageHandler(
        registry=registry, messages=message_handler.LOG_MESSAGE_HANDLERS
    )
    registry.register_service(message_handler.MessageHandler, log_message_handler)

    return registry
