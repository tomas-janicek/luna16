import typing

import pydantic
import torch

from luna16 import enums, hyperparameters_container, modules, services

from . import configurations


class ModelFactory:
    def __init__(self, registry: services.ServiceContainer) -> None:
        self.registry = registry

    def add_model(self, model_type: configurations.ModelType) -> typing.Self:
        hyperparameters = self.registry.get_service(
            hyperparameters_container.HyperparameterContainer
        )
        match model_type:
            case configurations.BiasesModel(n_blocks=n_blocks):
                module = modules.LunaModel(
                    in_channels=1,
                    conv_channels=8,
                    out_features=2,
                    n_blocks=n_blocks,
                    input_dim=(32, 48, 48),
                )
                hyperparameters.add_hyperparameter("model/n_block", n_blocks)
            case configurations.DropoutModel(
                n_blocks=n_blocks, dropout_rate=dropout_rate
            ):
                module = modules.LunaDropoutModel(
                    in_channels=1,
                    conv_channels=8,
                    out_features=2,
                    n_blocks=n_blocks,
                    input_dim=(32, 48, 48),
                    dropout_rate=dropout_rate,
                )
                hyperparameters.add_hyperparameter("model/n_block", n_blocks)
                hyperparameters.add_hyperparameter("model/dropout_rate", dropout_rate)
            case configurations.CnnLoadedModel(
                n_blocks=n_blocks,
                name=from_name,
                version=from_version,
                finetune=finetune,
                model_loader=model_loader,
            ):
                module = self.load_module(
                    loader=model_loader,
                    name=from_name,
                    version=from_version,
                    module_class=modules.LunaModel,
                    module_params=modules.LunaParameters(
                        in_channels=1,
                        conv_channels=8,
                        out_features=2,
                        n_blocks=n_blocks,
                        input_dim=(32, 48, 48),
                    ),
                )
                if finetune:
                    self.prepare_for_fine_tuning_head(module)
            case _:
                raise ValueError(f"Model type {model_type} not supported")

        hyperparameters.add_hyperparameter("model", module.__class__.__name__)

        self.registry.register_service(services.ClassificationModel, module)
        return self

    def add_optimizer(
        self,
        optimizer_type: configurations.OptimizerType,
    ) -> typing.Self:
        hyperparameters = self.registry.get_service(
            hyperparameters_container.HyperparameterContainer
        )
        module = self.registry.get_service(services.ClassificationModel)
        match optimizer_type:
            case configurations.AdamOptimizer(
                lr=lr, weight_decay=weight_decay, betas=betas
            ):
                optimizer = torch.optim.AdamW(
                    module.parameters(), lr=lr, weight_decay=weight_decay, betas=betas
                )
                hyperparameters.add_hyperparameter("optimizer/lr", lr)
                hyperparameters.add_hyperparameter(
                    "optimizer/weight_decay", weight_decay
                )
                hyperparameters.add_hyperparameter("optimizer/betas", betas)
            case configurations.SgdOptimizer(
                lr=lr, weight_decay=weight_decay, momentum=momentum
            ):
                optimizer = torch.optim.SGD(
                    module.parameters(),
                    lr=lr,
                    weight_decay=weight_decay,
                    momentum=momentum,
                )
                hyperparameters.add_hyperparameter("optimizer/lr", lr)
                hyperparameters.add_hyperparameter(
                    "optimizer/weight_decay", weight_decay
                )
                hyperparameters.add_hyperparameter("optimizer/momentum", momentum)
            case _:
                raise ValueError(f"Optimizer type {optimizer_type} not supported")

        hyperparameters.add_hyperparameter("optimizer", optimizer.__class__.__name__)
        self.registry.register_service(services.ClassificationOptimizer, optimizer)
        return self

    def add_scheduler(
        self, scheduler_type: configurations.SchedulerType
    ) -> typing.Self:
        hyperparameters = self.registry.get_service(
            hyperparameters_container.HyperparameterContainer
        )
        optimizer = self.registry.get_service(services.ClassificationOptimizer)
        match scheduler_type:
            case configurations.StepScheduler(gamma=gamma):
                lr_scheduler = torch.optim.lr_scheduler.StepLR(
                    optimizer, step_size=1, gamma=gamma
                )
                hyperparameters.add_hyperparameter("lr_scheduler/gamma", gamma)
            case _:
                raise ValueError(f"Scheduler type {scheduler_type} not supported")

        hyperparameters.add_hyperparameter(
            "lr_scheduler", lr_scheduler.__class__.__name__
        )
        self.registry.register_service(services.ClassificationScheduler, lr_scheduler)
        return self

    def load_module(
        self,
        loader: enums.ModelLoader,
        name: str,
        version: str,
        module_class: type[torch.nn.Module],
        module_params: pydantic.BaseModel,
    ) -> torch.nn.Module:
        match loader:
            case enums.ModelLoader.ML_FLOW:
                model_loader = self.registry.get_service(services.MLFlowModelSaver)
            case enums.ModelLoader.FILE:
                model_loader = self.registry.get_service(services.ModelSaver)

        module = model_loader.load_model(
            name=name,
            version=version,
            module_class=module_class,
            module_params=module_params,
        )
        return module

    def prepare_for_fine_tuning_head(self, module: torch.nn.Module) -> None:
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
