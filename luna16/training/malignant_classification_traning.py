import typing
from functools import partial

import torch
from ray import tune
from torch import nn

from luna16 import (
    batch_iterators,
    datasets,
    dto,
    message_handler,
    models,
    modules,
    services,
)
from luna16.modules.nodule_classfication.models import LunaModel
from luna16.training import trainers


class LunaMalignantClassificationLauncher:
    def __init__(
        self,
        validation_stride: int,
        state_name: str,
        state_version: str,
        training_name: str,
        registry: "services.ServiceContainer",
        training_length: int | None = None,
    ) -> None:
        self.validation_stride = validation_stride
        self.state_name = state_name
        self.state_version = state_version
        self.training_name = training_name
        self.registry = registry
        self.training_length = training_length
        self.logger = registry.get_service(message_handler.MessageHandler)
        self.batch_iterator = batch_iterators.BatchIteratorProvider(logger=self.logger)

    def fit(
        self,
        epochs: int,
        batch_size: int,
        version: str,
    ) -> dto.Scores:
        module = self._prepare_for_fine_tunning_head(
            name=self.state_name, version=self.state_version
        )
        model = models.NoduleClassificationModel(
            model=module,
            optimizer=torch.optim.SGD(module.parameters(), lr=0.001, momentum=0.99),
            batch_iterator=self.batch_iterator,
            logger=self.logger,
        )
        trainer = trainers.Trainer[dto.LunaClassificationCandidate](
            name=self.training_name, version=version, logger=self.logger
        )
        train, validation = datasets.create_pre_configured_luna_malignant(
            validation_stride=self.validation_stride,
        )
        data_module = datasets.DataModule(
            batch_size=batch_size,
            train=train,
            validation=validation,
        )
        return trainer.fit(model=model, epochs=epochs, data_module=data_module)

    def tune_parameters(
        self,
        epochs: int,
    ) -> tune.ResultGrid:
        hyperparameters: dict[str, typing.Any] = {
            "batch_size": tune.grid_search([16, 32, 64]),
            "learning_rate": tune.grid_search([0.0001, 0.001, 0.01]),
            "momentum": tune.grid_search([0.97, 0.98, 0.99]),
        }
        self.tunable_fit = partial(
            self.fit,
            epochs=epochs,
        )
        tuner = tune.Tuner(self.tunable_fit, param_space=hyperparameters)
        return tuner.fit()

    def _prepare_for_fine_tunning_head(self, name: str, version: str) -> nn.Module:
        model_saver = self.registry.get_service(services.ModelSaver)
        module = model_saver.load_model(
            module=LunaModel(),
            name=name,
            version=version,
        )

        # Create new state dict by taking everything from loaded state dict
        # except last block (the final linear part). Starting from a fully
        # initialized model would have us begin with (almost) all nodules
        # labeled as malignant, because that output means “nodule” in the
        # classifier we start from.
        module.luna_head = modules.LunaHead()

        finetune_blocks = ("luna_head",)

        # We to gather gradient only for blocks we will be fine-tuning.
        # This results in training only modifying parameters of finetune blocks.
        for name, parameter in module.named_parameters():
            if name.split(".")[0] not in finetune_blocks:
                parameter.requires_grad_(False)

        return module
