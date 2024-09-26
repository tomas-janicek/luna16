from functools import partial

import torch
from ray import tune

from luna16 import (
    batch_iterators,
    datasets,
    dto,
    models,
    modules,
    services,
    trainers,
)


class LunaClassificationLauncher:
    def __init__(
        self,
        training_name: str,
        validation_stride: int,
        num_workers: int,
        registry: services.ServiceContainer,
        validation_cadence: int,
        training_length: int | None = None,
    ) -> None:
        self.training_name = training_name
        self.validation_stride = validation_stride
        self.validation_cadence = validation_cadence
        self.num_workers = num_workers
        self.training_length = training_length
        self.registry = registry
        self.logger = registry.get_service(services.LogMessageHandler)
        self.hyperparameters = registry.get_service(services.Hyperparameters)
        self.batch_iterator = batch_iterators.BatchIteratorProvider(logger=self.logger)

    def fit(
        self,
        *,
        epochs: int,
        batch_size: int,
        lr: float,
        momentum: float,
        conv_channels: int,
    ) -> dto.Scores:
        module = modules.LunaModel(
            in_channels=1,
            conv_channels=conv_channels,
        )
        model = models.NoduleClassificationModel(
            model=module,
            optimizer=torch.optim.SGD(module.parameters(), lr=lr, momentum=momentum),
            batch_iterator=self.batch_iterator,
            logger=self.logger,
            validation_cadence=self.validation_cadence,
        )
        trainer = trainers.Trainer(name=self.training_name, logger=self.logger)
        train, validation = datasets.create_pre_configured_luna_rationed(
            validation_stride=self.validation_stride,
            training_length=self.training_length,
        )
        data_module = datasets.DataModule(
            batch_size=batch_size,
            num_workers=self.num_workers,
            train=train,
            validation=validation,
        )
        return trainer.fit(model=model, epochs=epochs, data_module=data_module)

    def tune_parameters(
        self,
        epochs: int,
    ) -> tune.ResultGrid:
        hyperparameters = {
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
