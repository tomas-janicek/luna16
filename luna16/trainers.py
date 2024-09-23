import typing
from datetime import datetime

from luna16 import datasets, models, training_logging

CandidateT = typing.TypeVar("CandidateT")


class BaseTrainer(typing.Protocol[CandidateT]):
    def fit(
        self,
        *,
        model: models.BaseModel[CandidateT],
        epochs: int,
        data_module: datasets.DataModule[CandidateT],
    ) -> None: ...

    def fit_epoch(
        self,
        *,
        epoch: int,
        epochs: int,
        model: models.BaseModel[CandidateT],
        data_module: datasets.DataModule[CandidateT],
    ) -> None: ...


class Trainer(BaseTrainer[CandidateT]):
    def __init__(
        self,
        name: str,
        logger: training_logging.LogMessageHandler,
    ) -> None:
        self.name = name
        self.logger = logger

    def fit(
        self,
        *,
        model: models.BaseModel[CandidateT],
        epochs: int,
        data_module: datasets.DataModule[CandidateT],
    ) -> None:
        self.logger.registry.call_all_creators(
            training_name=self.name, training_start_time=datetime.now()
        )
        log_start_training = training_logging.LogStart(training_description=str(model))
        self.logger.handle_message(log_start_training)

        for epoch in range(1, epochs + 1):
            self.fit_epoch(
                epoch=epoch,
                epochs=epochs,
                model=model,
                data_module=data_module,
            )

        log_model = training_logging.LogModel(
            model=model.get_module(),
            training_name=self.name,
            signature=model.get_signature(
                train_dl=data_module.get_training_dataloader()
            ),
        )
        self.logger.handle_message(log_model)

    def fit_epoch(
        self,
        *,
        epoch: int,
        epochs: int,
        model: models.BaseModel[CandidateT],
        data_module: datasets.DataModule[CandidateT],
    ) -> None:
        log_epoch = training_logging.LogEpoch(
            epoch=epoch,
            n_epochs=epochs,
            batch_size=data_module.batch_size,
            training_length=data_module.training_len,
            validation_length=data_module.validation_len,
        )
        self.logger.handle_message(log_epoch)

        train_dl = data_module.get_training_dataloader()
        validation_dl = data_module.get_validation_dataloader()
        model.fit_epoch(epoch=epoch, train_dl=train_dl, validation_dl=validation_dl)

    def __repr__(self) -> str:
        _repr = (
            f"{self.__class__.__name__}("
            f"name={self.name}, "
            f"logger={self.logger.__class__.__name__})"
        )
        return _repr
