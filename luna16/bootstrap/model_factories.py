import typing

import pydantic
import torch

from luna16 import enums, hyperparameters_container, modules, services


class ModelFactory:
    def __init__(self, registry: services.ServiceContainer) -> None:
        self.registry = registry

    def add_model(self, model_type: enums.ModelType) -> typing.Self:
        hyperparameters = self.registry.get_service(
            hyperparameters_container.HyperparameterContainer
        )
        match model_type:
            case enums.ConvModel():
                n_blocks = 4
                module = modules.LunaModel(
                    in_channels=1,
                    conv_channels=8,
                    out_features=2,
                    n_blocks=n_blocks,
                    input_dim=(32, 48, 48),
                )
                hyperparameters.add_hyperparameter("model/n_block", n_blocks)
            case enums.DropoutModel():
                n_blocks = 4
                dropout_rate = 0.15
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
            case enums.ConvLoadedModel(
                name=from_name,
                version=from_version,
                finetune=finetune,
                model_loader=model_loader,
            ):
                n_blocks = 4
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
        optimizer_type: enums.OptimizerType,
    ) -> typing.Self:
        hyperparameters = self.registry.get_service(
            hyperparameters_container.HyperparameterContainer
        )
        module = self.registry.get_service(services.ClassificationModel)
        match optimizer_type:
            case enums.OptimizerType.ADAM:
                lr = 1e-3
                weight_decay = 1e-2
                betas = (0.9, 0.999)
                optimizer = torch.optim.AdamW(
                    module.parameters(), lr=lr, weight_decay=weight_decay, betas=betas
                )
                hyperparameters.add_hyperparameter("optimizer/lr", lr)
                hyperparameters.add_hyperparameter(
                    "optimizer/weight_decay", weight_decay
                )
                hyperparameters.add_hyperparameter("optimizer/betas", betas)
            case enums.OptimizerType.SLOWER_ADAM:
                lr = 1e-3
                weight_decay = 1e-4
                betas = (0.9, 0.999)
                optimizer = torch.optim.AdamW(
                    module.parameters(), lr=lr, weight_decay=weight_decay, betas=betas
                )
                hyperparameters.add_hyperparameter("optimizer/lr", lr)
                hyperparameters.add_hyperparameter(
                    "optimizer/weight_decay", weight_decay
                )
                hyperparameters.add_hyperparameter("optimizer/betas", betas)

        hyperparameters.add_hyperparameter("optimizer", optimizer.__class__.__name__)
        self.registry.register_service(services.ClassificationOptimizer, optimizer)
        return self

    def add_scheduler(self, scheduler_type: enums.SchedulerType) -> typing.Self:
        hyperparameters = self.registry.get_service(
            hyperparameters_container.HyperparameterContainer
        )
        optimizer = self.registry.get_service(services.ClassificationOptimizer)
        match scheduler_type:
            case enums.SchedulerType.STEP:
                gamma = 0.90
                lr_scheduler = torch.optim.lr_scheduler.StepLR(
                    optimizer, step_size=1, gamma=gamma
                )
                hyperparameters.add_hyperparameter("lr_scheduler/gamma", gamma)
            case enums.SchedulerType.SLOWER_STEP:
                gamma = 0.1
                lr_scheduler = torch.optim.lr_scheduler.StepLR(
                    optimizer, step_size=1, gamma=gamma
                )
                hyperparameters.add_hyperparameter("lr_scheduler/gamma", gamma)

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
