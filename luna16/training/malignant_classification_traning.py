import typing
from functools import partial

import torch
from ray import tune

from luna16 import (
    batch_iterators,
    datasets,
    dto,
    message_handler,
    models,
    trainers,
)
from luna16.modules.nodule_classfication.models import LunaModel

from . import model_saver

if typing.TYPE_CHECKING:
    from luna16 import services


class LunaMalignantClassificationLauncher:
    def __init__(
        self,
        validation_stride: int,
        state_name: str,
        training_name: str,
        registry: "services.ServiceContainer",
        training_length: int | None = None,
    ) -> None:
        self.validation_stride = validation_stride
        self.state_name = state_name
        self.training_name = training_name
        self.registry = registry
        self.training_length = training_length
        self.logger = registry.get_service(message_handler.MessageHandler)
        self.batch_iterator = batch_iterators.BatchIteratorProvider(logger=self.logger)

    def fit(
        self,
        epochs: int,
        batch_size: int,
    ) -> dto.Scores:
        classification_model_saver = model_saver.ModelSaver(
            model_name="classification",
        )
        module = classification_model_saver.load_model(
            model=LunaModel(), state_name=self.state_name, n_excluded_blocks=2
        )
        model = models.NoduleClassificationModel(
            model=module,
            optimizer=torch.optim.SGD(module.parameters(), lr=0.001, momentum=0.99),
            batch_iterator=self.batch_iterator,
            logger=self.logger,
        )
        trainer = trainers.Trainer[dto.LunaClassificationCandidate](
            name=self.training_name, logger=self.logger
        )
        train, validation = datasets.create_pre_configured_luna_rationed(
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
