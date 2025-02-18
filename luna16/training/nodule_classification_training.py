import typing
from functools import partial

from ray import tune
from torch.profiler import schedule

from luna16 import (
    batch_iterators,
    datasets,
    dto,
    hyperparameters_container,
    message_handler,
    models,
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
        log_every_n_examples: int,
        profile: bool = False,
    ) -> dto.Scores:
        module = self.registry.get_service(services.ClassificationModel)
        optimizer = self.registry.get_service(services.ClassificationOptimizer)
        lr_scheduler = self.registry.get_service(services.ClassificationScheduler)
        model = models.NoduleClassificationModel(
            module=module,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
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

    # TODO: move this to CLI and connect it to custom bootstrap
    def tune_parameters(
        self,
        epochs: int,
    ) -> tune.ResultGrid:
        hyperparameters: dict[str, typing.Any] = {
            "batch_size": tune.grid_search([64, 128, 256]),
            "learning_rate": tune.grid_search([0.00001, 0.0001, 0.001]),
            "scheduler_gamma": tune.grid_search([0.1, 0.5, 0.9]),
            "weight_decay": tune.grid_search([0.0001, 0.001, 0.01]),
            "luna_blocks": tune.grid_search([4, 8, 16]),
            "dropout_rate": tune.grid_search([0.1, 0.25, 0.3, 0.35, 0.4]),
        }
        self.tunable_fit = partial(
            self.fit,
            epochs=epochs,
        )
        tuner = tune.Tuner(self.tunable_fit, param_space=hyperparameters)
        return tuner.fit()
