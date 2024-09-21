import datetime
import socket
import typing
from pathlib import Path

import mlflow
import torch
from torch import nn
from torch.utils import data as data_utils
from torch.utils.tensorboard.writer import SummaryWriter

from luna16 import enums
from luna16.settings import settings
from luna16.training_logging import ct_image_logger_wrapper

from .. import dto
from . import base, logger_wrappers
from . import dto as training_dto

T = typing.TypeVar("T")


class ClassificationLoggingAdapter:
    def __init__(self, training_name: str) -> None:
        self.training_name = training_name
        self.tensor_board_logger = logger_wrappers.TensorBoardLoggerWrapper(
            training_name=training_name
        )
        self.console_logger = logger_wrappers.ConsoleLoggerWrapper()
        self.ml_flow_logger = logger_wrappers.MlFlowLoggerWrapper(
            experiment_name=training_name,
        )
        self.all_loggers: list[base.BaseLoggerWrapper] = [
            self.console_logger,
            self.ml_flow_logger,
            self.tensor_board_logger,
        ]
        self.metrics_loggers: list[
            base.MetricsLoggerWrapper[training_dto.NumberValue]
        ] = [
            self.console_logger,
            self.ml_flow_logger,
            self.tensor_board_logger,
        ]
        self.results_loggers: list[base.ResultLoggerWrapper] = [
            self.tensor_board_logger
        ]
        self.training_process_loggers: list[base.TrainingProgressLoggerWrapper] = [
            self.console_logger
        ]
        self.batch_loggers: list[base.BatchLoggerWrapper] = [self.console_logger]

    def log_start_training(
        self,
        *,
        training_api: typing.Any,
        n_epochs: int,
        batch_size: int,
        train_dl: data_utils.DataLoader[T],
        validation_dl: data_utils.DataLoader[T],
    ) -> None:
        # Initialize TensorBoard for logging metrics
        current_time = datetime.datetime.now().isoformat()
        hostname = socket.gethostname()
        training_log_dir = Path(
            f"runs/{self.training_name}/{current_time}_{hostname}-training"
        )
        validation_log_dir = Path(
            f"runs/{self.training_name}/{current_time}_{hostname}-validation"
        )
        training_writer = SummaryWriter(log_dir=training_log_dir)
        validation_writer = SummaryWriter(log_dir=validation_log_dir)

        mlflow.set_tracking_uri(uri=settings.ML_FLOW_URL)
        experiment = mlflow.set_experiment(experiment_name=self.training_name)
        active_run = mlflow.start_run(experiment_id=experiment.experiment_id)

        self.console_logger.open_logger(
            training_api=training_api,
            n_epochs=n_epochs,
            batch_size=batch_size,
            train_dl=train_dl,
            validation_dl=validation_dl,
        )
        self.tensor_board_logger.open_logger(
            training_writer=training_writer,
            validation_writer=validation_writer,
        )
        self.ml_flow_logger.open_logger(
            experiment=experiment,
            active_run=active_run,
        )

        for logger in self.training_process_loggers:
            logger.log_start()

    def log_epoch(self, *, epoch: int) -> None:
        for logger in self.training_process_loggers:
            logger.log_epoch(epoch=epoch)

    def log_metrics(
        self,
        *,
        values: dict[str, training_dto.NumberValue],
        epoch: int,
        n_processed_samples: int,
        mode: enums.Mode,
    ) -> None:
        for logger in self.metrics_loggers:
            logger.log_metrics(
                values=values,
                epoch=epoch,
                n_processed_samples=n_processed_samples,
                mode=mode,
            )

    def log_results(
        self,
        *,
        epoch: int,
        mode: enums.Mode,
        n_processed_samples: int,
        labels: torch.Tensor,
        predictions: torch.Tensor,
    ) -> None:
        for logger in self.results_loggers:
            logger.log_result(
                epoch=epoch,
                mode=mode,
                n_processed_samples=n_processed_samples,
                labels=labels,
                predictions=predictions,
            )

    def close_all(self) -> None:
        for logger in self.all_loggers:
            logger.close_logger()


class SegmentationLoggingAdapter:
    def __init__(self, training_name: str) -> None:
        self.training_name = training_name
        self.tensor_board_logger = logger_wrappers.TensorBoardLoggerWrapper(
            training_name=self.training_name
        )
        self.console_logger = logger_wrappers.ConsoleLoggerWrapper()
        self.ml_flow_logger = logger_wrappers.MlFlowLoggerWrapper(
            experiment_name=self.training_name
        )
        self.ct_image_logger = ct_image_logger_wrapper.CtImageLoggerWrapper(
            training_name=self.training_name
        )

        self.all_loggers = [
            self.console_logger,
            self.ml_flow_logger,
            self.tensor_board_logger,
            self.ct_image_logger,
        ]
        self.training_process_loggers: list[base.TrainingProgressLoggerWrapper] = [
            self.console_logger
        ]
        self.metrics_loggers: list[
            base.MetricsLoggerWrapper[training_dto.NumberValue]
        ] = [
            self.console_logger,
            self.ml_flow_logger,
            self.tensor_board_logger,
        ]
        self.batch_loggers: list[base.BatchLoggerWrapper] = [self.console_logger]

    def log_start_training(
        self,
        *,
        training_api: typing.Any,
        n_epochs: int,
        batch_size: int,
        train_dl: data_utils.DataLoader[T],
        validation_dl: data_utils.DataLoader[T],
    ) -> None:
        # Initialize TensorBoard for logging metrics
        current_time = datetime.datetime.now().isoformat()
        hostname = socket.gethostname()
        training_log_dir = Path(
            f"runs/{self.training_name}/{current_time}_{hostname}-training"
        )
        validation_log_dir = Path(
            f"runs/{self.training_name}/{current_time}_{hostname}-validation"
        )
        training_writer = SummaryWriter(log_dir=training_log_dir)
        validation_writer = SummaryWriter(log_dir=validation_log_dir)

        mlflow.set_tracking_uri(uri=settings.ML_FLOW_URL)
        experiment = mlflow.set_experiment(experiment_name=self.training_name)
        active_run = mlflow.start_run(experiment_id=experiment.experiment_id)

        self.console_logger.open_logger(
            training_api=training_api,
            n_epochs=n_epochs,
            batch_size=batch_size,
            train_dl=train_dl,
            validation_dl=validation_dl,
        )
        self.tensor_board_logger.open_logger(
            training_writer=training_writer,
            validation_writer=validation_writer,
        )
        self.ct_image_logger.open_logger(
            training_writer=training_writer,
            validation_writer=validation_writer,
        )
        self.ml_flow_logger.open_logger(
            experiment=experiment,
            active_run=active_run,
        )

        for logger in self.training_process_loggers:
            logger.log_start()

    def log_epoch(self, *, epoch: int) -> None:
        for logger in self.training_process_loggers:
            logger.log_epoch(epoch=epoch)

    def log_metrics(
        self,
        *,
        values: dict[str, training_dto.NumberValue],
        epoch: int,
        n_processed_samples: int,
        mode: enums.Mode,
    ) -> None:
        for logger in self.metrics_loggers:
            logger.log_metrics(
                values=values,
                epoch=epoch,
                n_processed_samples=n_processed_samples,
                mode=mode,
            )

    def log_images(
        self,
        *,
        epoch: int,
        mode: enums.Mode,
        n_processed_samples: int,
        dataloader: data_utils.DataLoader[dto.LunaSegmentationCandidate],
        model: nn.Module,
        device: torch.device,
    ) -> None:
        self.ct_image_logger.log_images(
            epoch=epoch,
            mode=mode,
            n_processed_samples=n_processed_samples,
            dataloader=dataloader,
            model=model,
            device=device,
        )

    def close_all(self) -> None:
        for logger in self.all_loggers:
            logger.close_logger()
