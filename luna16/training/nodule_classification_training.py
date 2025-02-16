import typing
from functools import partial

import torch
from ray import tune
from torch.profiler import schedule

from luna16 import (
    batch_iterators,
    datasets,
    dto,
    enums,
    hyperparameters_container,
    message_handler,
    models,
    modules,
    services,
)

from . import trainers


class LunaClassificationLauncher:
    def __init__(
        self,
        training_name: str,
        validation_stride: int,
        registry: services.ServiceContainer,
        validation_cadence: int,
    ) -> None:
        self.training_name = training_name
        self.validation_stride = validation_stride
        self.validation_cadence = validation_cadence
        self.registry = registry
        self.logger = registry.get_service(message_handler.MessageHandler)
        self.hyperparameters = registry.get_service(
            hyperparameters_container.HyperparameterContainer
        )
        self.batch_iterator = batch_iterators.BatchIteratorProvider(logger=self.logger)

    def fit(
        self,
        *,
        version: str,
        epochs: int,
        batch_size: int,
        lr: float,
        momentum: float,
        log_every_n_examples: int,
        weight_decay: float = 1e-2,
        profile: bool = False,
    ) -> dto.Scores:
        module = modules.LunaModel()
        _sgd = torch.optim.SGD(
            module.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay
        )
        _adam = torch.optim.Adam(module.parameters(), lr=lr, weight_decay=weight_decay)
        model = models.NoduleClassificationModel(
            module=module,
            optimizer=_adam,
            batch_iterator=self.batch_iterator,
            logger=self.logger,
            validation_cadence=self.validation_cadence,
            log_every_n_examples=log_every_n_examples,
        )
        trainer = trainers.Trainer[dto.LunaClassificationCandidate](
            name=self.training_name, version=version, logger=self.logger
        )
        train, validation = datasets.create_pre_configured_luna_cutouts(
            validation_stride=self.validation_stride
        )
        data_module = datasets.DataModule(
            batch_size=batch_size,
            train=train,
            validation=validation,
        )
        if profile:
            tracing_schedule = schedule(
                skip_first=1, wait=1, warmup=1, active=1, repeat=4
            )
            return trainer.fit_profile(
                model=model,
                epochs=epochs,
                data_module=data_module,
                tracing_schedule=tracing_schedule,
            )
        return trainer.fit(model=model, epochs=epochs, data_module=data_module)

    def load_fit(
        self,
        *,
        version: str,
        epochs: int,
        batch_size: int,
        from_saver: enums.ModelLoader,
        from_name: str,
        from_version: str,
        lr: float,
        momentum: float,
        profile: bool = False,
        log_every_n_examples: int,
        finetune: bool = False,
    ) -> dto.Scores:
        module = trainers.load_module(
            registry=self.registry,
            loader=from_saver,
            name=from_name,
            version=from_version,
            module_class=modules.LunaModel,
        )

        model = models.NoduleClassificationModel(
            module=module,
            optimizer=torch.optim.SGD(module.parameters(), lr=lr, momentum=momentum),
            batch_iterator=self.batch_iterator,
            logger=self.logger,
            log_every_n_examples=log_every_n_examples,
        )
        if finetune:
            model.prepare_for_fine_tuning_head()
        trainer = trainers.Trainer[dto.LunaClassificationCandidate](
            name=self.training_name, version=version, logger=self.logger
        )
        train, validation = datasets.create_pre_configured_luna_cutouts(
            validation_stride=self.validation_stride,
        )
        data_module = datasets.DataModule(
            batch_size=batch_size,
            train=train,
            validation=validation,
        )
        if profile:
            tracing_schedule = schedule(
                skip_first=1, wait=1, warmup=1, active=1, repeat=4
            )
            return trainer.fit_profile(
                model=model,
                epochs=epochs,
                data_module=data_module,
                tracing_schedule=tracing_schedule,
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
