import time
import typing
from datetime import datetime

from torch.profiler import profile

from luna16 import (
    datasets,
    message_handler,
    models,
    scoring,
    settings,
    utils,
)

CandidateT = typing.TypeVar("CandidateT")


class BaseTrainer(typing.Protocol[CandidateT]):
    def fit(
        self,
        *,
        model: models.BaseModel[CandidateT],
        epochs: int,
        data_module: datasets.DataModule[CandidateT],
    ) -> scoring.PerformanceMetrics: ...

    def fit_epoch(
        self,
        *,
        epoch: int,
        epochs: int,
        model: models.BaseModel[CandidateT],
        data_module: datasets.DataModule[CandidateT],
    ) -> scoring.PerformanceMetrics: ...


class Trainer(BaseTrainer[CandidateT]):
    def __init__(
        self,
        name: str,
        version: str,
        logger: message_handler.BaseMessageHandler,
    ) -> None:
        self.name = name
        self.version = version
        self.message_handler = logger

    def fit(
        self,
        *,
        model: models.BaseModel[CandidateT],
        epochs: int,
        data_module: datasets.DataModule[CandidateT],
    ) -> scoring.PerformanceMetrics:
        training_start_time = datetime.now()
        self.message_handler.registry.call_all_creators(
            training_name=self.name, training_start_time=training_start_time
        )
        self.message_handler.handle_message(message_handler.LogParams())
        log_start_training = message_handler.LogStart(training_description=str(model))
        self.message_handler.handle_message(log_start_training)

        score = None
        start_time = time.time()
        for epoch in range(1, epochs + 1):
            score = self.fit_epoch(
                epoch=epoch,
                epochs=epochs,
                model=model,
                data_module=data_module,
            )

        if score is None:
            raise ValueError("No score was returned from training loop!")

        end_time = time.time()
        print(f"Training time: {end_time - start_time} seconds.")

        log_model = message_handler.LogModel(
            module=model.get_module(),
            training_name=self.name,
            signature=model.get_signature(
                train_dl=data_module.get_training_dataloader()
            ),
            version=self.version,
        )
        self.message_handler.handle_message(log_model)
        return score

    def fit_profile(
        self,
        *,
        model: models.BaseModel[CandidateT],
        epochs: int,
        data_module: datasets.DataModule[CandidateT],
        tracing_schedule: typing.Callable[..., typing.Any],
    ) -> scoring.PerformanceMetrics:
        if epochs < 2:
            raise ValueError("Profiling requires at least two epochs!")

        training_start_time = datetime.now()
        self.message_handler.registry.call_all_creators(
            training_name=self.name, training_start_time=training_start_time
        )
        log_start_training = message_handler.LogStart(training_description=str(model))
        self.message_handler.handle_message(log_start_training)

        score = None
        with profile(
            schedule=tracing_schedule,
            record_shapes=True,
            profile_memory=True,
            with_modules=True,
        ) as prof:
            start_time = time.time()
            for epoch in range(1, epochs + 1):
                score = self.fit_epoch(
                    epoch=epoch,
                    epochs=epochs,
                    model=model,
                    data_module=data_module,
                )
                prof.step()
            end_time = time.time()

        if score is None:
            raise ValueError("No score was returned from training loop!")

        print(f"Training time: {end_time - start_time} seconds.")
        prof.export_chrome_trace(
            str(
                settings.PROFILING_DIR
                / f"trace_{self.name.lower()}_{utils.get_datetime_string(training_start_time)}.json"
            )
        )

        log_model = message_handler.LogModel(
            module=model.get_module(),
            training_name=self.name,
            signature=model.get_signature(
                train_dl=data_module.get_training_dataloader()
            ),
            version=self.version,
        )
        self.message_handler.handle_message(log_model)
        return score

    def fit_epoch(
        self,
        *,
        epoch: int,
        epochs: int,
        model: models.BaseModel[CandidateT],
        data_module: datasets.DataModule[CandidateT],
    ) -> scoring.PerformanceMetrics:
        log_epoch = message_handler.LogEpoch(
            epoch=epoch,
            n_epochs=epochs,
            batch_size=data_module.batch_size,
            training_length=data_module.training_len,
            validation_length=data_module.validation_len,
        )
        self.message_handler.handle_message(log_epoch)

        train_dl = data_module.get_training_dataloader()
        validation_dl = data_module.get_validation_dataloader()
        return model.fit_epoch(
            epoch=epoch, train_dl=train_dl, validation_dl=validation_dl
        )

    def __repr__(self) -> str:
        _repr = (
            f"{self.__class__.__name__}("
            f"name={self.name}, "
            f"logger={self.message_handler.__class__.__name__})"
        )
        return _repr
