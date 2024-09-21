import typing

from luna16 import datasets, models, training_logging

from . import base

CandidateT = typing.TypeVar("CandidateT")


class Trainer(base.BaseTrainer[CandidateT]):
    def __init__(
        self,
        logger: training_logging.ClassificationLoggingAdapter
        | training_logging.SegmentationLoggingAdapter,
    ) -> None:
        self.classification_logger = logger

    def fit(
        self,
        *,
        model: models.BaseModel[CandidateT],
        epochs: int,
        data_module: datasets.DataModule[CandidateT],
    ) -> None:
        self.classification_logger.log_start_training(
            training_api=self,
            n_epochs=epochs,
            batch_size=data_module.batch_size,
            train_dl=data_module.get_training_dataloader(),
            validation_dl=data_module.get_validation_dataloader(),
        )

        for epoch in range(1, epochs + 1):
            self.fit_epoch(
                model=model,
                epoch=epoch,
                data_module=data_module,
            )

    def fit_epoch(
        self,
        *,
        model: models.BaseModel[CandidateT],
        epoch: int,
        data_module: datasets.DataModule[CandidateT],
    ) -> None:
        self.classification_logger.log_epoch(epoch=epoch)
        train_dl = data_module.get_training_dataloader()
        validation_dl = data_module.get_validation_dataloader()

        model.fit_epoch(epoch=epoch, train_dl=train_dl, validation_dl=validation_dl)
