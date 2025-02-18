import pydantic
import torch

from luna16 import enums, message_handler, modules, services
from luna16.hyperparameters_container import HyperparameterContainer


def create_registry(
    model_type: enums.ModelType,
    optimizer_type: enums.OptimizerType = enums.OptimizerType.ADAM,
    scheduler_type: enums.SchedulerType = enums.SchedulerType.STEP,
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

    registry.register_service(
        type=services.MLFlowModelSaver,
        value=services.MLFlowModelSaver(),
    )
    registry.register_service(
        type=services.ModelSaver,
        value=services.ModelSaver(),
    )

    hyperparameters = HyperparameterContainer()
    registry.register_service(HyperparameterContainer, hyperparameters)

    log_message_handler = message_handler.MessageHandler(
        registry=registry, messages=message_handler.LOG_MESSAGE_HANDLERS
    )
    registry.register_service(message_handler.MessageHandler, log_message_handler)

    match model_type:
        case enums.ConvModel():
            module = modules.LunaModel(
                in_channels=1,
                conv_channels=8,
                out_features=2,
                n_blocks=4,
                input_dim=(32, 48, 48),
            )
        case enums.DropoutModel():
            module = modules.LunaDropoutModel(
                in_channels=1,
                conv_channels=8,
                out_features=2,
                n_blocks=4,
                input_dim=(32, 48, 48),
            )
        case enums.ConvLoadedModel(
            name=from_name,
            version=from_version,
            finetune=finetune,
            model_loader=model_loader,
        ):
            module = load_module(
                registry=registry,
                loader=model_loader,
                name=from_name,
                version=from_version,
                module_class=modules.LunaModel,
                module_params=modules.LunaParameters(
                    in_channels=1,
                    conv_channels=8,
                    out_features=2,
                    n_blocks=4,
                    input_dim=(32, 48, 48),
                ),
            )
            if finetune:
                prepare_for_fine_tuning_head(module)

        case _:
            raise ValueError(f"Model type {model_type} not supported")

    registry.register_service(services.ClassificationModel, module)

    match optimizer_type:
        case enums.OptimizerType.ADAM:
            optimizer = torch.optim.AdamW(
                module.parameters(), lr=1e-3, weight_decay=1e-2, betas=(0.9, 0.999)
            )
        case enums.OptimizerType.SLOWER_ADAM:
            optimizer = torch.optim.AdamW(
                module.parameters(), lr=1e-3, weight_decay=1e-4, betas=(0.9, 0.999)
            )

    registry.register_service(services.ClassificationOptimizer, optimizer)

    match scheduler_type:
        case enums.SchedulerType.STEP:
            lr_scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=1, gamma=0.90
            )
        case enums.SchedulerType.SLOWER_STEP:
            lr_scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=1, gamma=0.1
            )

    registry.register_service(services.ClassificationScheduler, lr_scheduler)

    return registry


def load_module(
    registry: services.ServiceContainer,
    loader: enums.ModelLoader,
    name: str,
    version: str,
    module_class: type[torch.nn.Module],
    module_params: pydantic.BaseModel,
) -> torch.nn.Module:
    match loader:
        case enums.ModelLoader.ML_FLOW:
            model_loader = registry.get_service(services.MLFlowModelSaver)
        case enums.ModelLoader.FILE:
            model_loader = registry.get_service(services.ModelSaver)

    module = model_loader.load_model(
        name=name,
        version=version,
        module_class=module_class,
        module_params=module_params,
    )
    return module


def prepare_for_fine_tuning_head(module: torch.nn.Module) -> None:
    # Replace only luna head. Starting from a fully
    # initialized model would have us begin with (almost) all nodules
    # labeled as malignant, because that output means “nodule” in the
    # classifier we start from.
    module.luna_head = modules.LunaHead(in_features=1152, out_features=2)

    finetune_blocks = ("luna_head",)

    # We to gather gradient only for blocks we will be fine-tuning.
    # This results in training only modifying parameters of finetune blocks.
    for name, parameter in module.named_parameters():
        if name.split(".")[0] not in finetune_blocks:
            parameter.requires_grad_(False)
