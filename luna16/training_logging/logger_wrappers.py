import datetime
import logging
import time
import typing

import mlflow
import torch
from mlflow.entities import Experiment
from torch.utils import data as data_utils
from torch.utils.tensorboard.writer import SummaryWriter

from luna16 import dto, enums

from . import base

_log = logging.getLogger(__name__)


T = typing.TypeVar("T")


class ConsoleLoggerWrapper(
    base.MetricsLoggerWrapper[dto.NumberValue],
    base.TrainingProgressLoggerWrapper,
    base.BatchLoggerWrapper,
):
    def open_logger(
        self,
        *,
        training_api: typing.Any,
        n_epochs: int,
        batch_size: int,
        train_dl: data_utils.DataLoader[T],
        validation_dl: data_utils.DataLoader[T],
        **kwargs: typing.Any,
    ) -> None:
        self.training_api = training_api
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.train_dl = train_dl
        self.validation_dl = validation_dl

    def log_start(self) -> None:
        _log.info(f"Starting {self.training_api}")

    def log_epoch(self, *, epoch: int) -> None:
        _log.info(
            f"E {epoch:04d} of {self.n_epochs:04d}, {len(self.train_dl)}/{len(self.validation_dl)} batches of "
            f"size {self.batch_size}"
        )

    def log_metrics(
        self,
        *,
        values: dict[str, dto.NumberValue],
        epoch: int,
        n_processed_samples: int,
        mode: enums.Mode,
    ) -> None:
        formatted_values = ", ".join(
            (
                f"{value.name.capitalize()}: {value.formatted_value}"
                for _, value in values.items()
            )
        )
        msg = f"E {epoch:04d} {mode.value:>10} " + formatted_values
        _log.info(msg)

    def log_bach_start(self, *, epoch: int, mode: enums.Mode, batch_size: int) -> None:
        _log.info(
            f"E {epoch:04d} {mode.value:>10} ----/{batch_size}, starting",
        )

    def log_batch(
        self,
        *,
        epoch: int,
        mode: enums.Mode,
        batch_size: int,
        batch_index: int,
        started_at: float,
    ) -> None:
        estimated_duration_in_seconds = (
            (time.time() - started_at) / (batch_index + 1) * (batch_size)
        )

        estimated_done_at = datetime.datetime.fromtimestamp(
            started_at + estimated_duration_in_seconds
        )
        estimated_duration = datetime.timedelta(seconds=estimated_duration_in_seconds)
        estimated_done_at_str = str(estimated_done_at).rsplit(".", 1)[0]
        estimated_duration_str = str(estimated_duration).rsplit(".", 1)[0]
        _log.info(
            f"E {epoch:04d} {mode.value:>10} {batch_index:-4}/{batch_size}, "
            f"done at {estimated_done_at_str}, {estimated_duration_str}"
        )

    def log_bach_end(self, *, epoch: int, mode: enums.Mode, batch_size: int) -> None:
        now_dt = str(datetime.datetime.now()).rsplit(".", 1)[0]
        _log.info(f"E {epoch:04d} {mode.value:>10} ----/{batch_size}, done at {now_dt}")

    def close_logger(self) -> None: ...


class TensorBoardLoggerWrapper(
    base.MetricsLoggerWrapper[dto.NumberValue], base.ResultLoggerWrapper
):
    def __init__(self, training_name: str) -> None:
        self.training_name = training_name

    def open_logger(
        self,
        *,
        training_writer: SummaryWriter,
        validation_writer: SummaryWriter,
        **kwargs: typing.Any,
    ) -> None:
        self.training_writer = training_writer
        self.validation_writer = validation_writer

    def log_metrics(
        self,
        *,
        epoch: int,
        mode: enums.Mode,
        n_processed_samples: int,
        values: dict[str, dto.NumberValue],
    ) -> None:
        tensorboard_writer = self._get_writer(mode=mode)
        for key, value in values.items():
            tensorboard_writer.add_scalar(
                tag=key,
                scalar_value=value.value,
                global_step=n_processed_samples,
            )

        tensorboard_writer.flush()

    def log_results(
        self,
        *,
        epoch: int,
        mode: enums.Mode,
        n_processed_samples: int,
        labels: torch.Tensor,
        predictions: torch.Tensor,
    ) -> None:
        tensorboard_writer = self._get_writer(mode=mode)

        negative_label_mask = labels == 0
        positive_label_mask = labels == 1

        # Precision-Recall Curves
        tensorboard_writer.add_pr_curve(
            tag="pr",
            labels=labels,
            predictions=predictions,
            global_step=n_processed_samples,
        )

        negative_histogram_mask = negative_label_mask & (predictions > 0.01)
        positive_histogram_mask = positive_label_mask & (predictions < 0.99)

        bins = [x / 50.0 for x in range(51)]
        if negative_histogram_mask.any():
            tensorboard_writer.add_histogram(
                tag="is_neg",
                values=predictions[negative_histogram_mask],
                global_step=n_processed_samples,
                bins=bins,  # type: ignore
            )
        if positive_histogram_mask.any():
            tensorboard_writer.add_histogram(
                tag="is_pos",
                values=predictions[positive_histogram_mask],
                global_step=n_processed_samples,
                bins=bins,  # type: ignore
            )

    def close_logger(self) -> None:
        self.training_writer.close()
        self.validation_writer.close()

    def _get_writer(self, mode: enums.Mode) -> SummaryWriter:
        match mode:
            case enums.Mode.TRAINING:
                tensorboard_writer = self.training_writer
            case enums.Mode.VALIDATING:
                tensorboard_writer = self.validation_writer
        return tensorboard_writer


class MlFlowLoggerWrapper(base.MetricsLoggerWrapper[dto.NumberValue]):
    def __init__(self, experiment_name: str) -> None:
        self.experiment_name = experiment_name

    def open_logger(
        self,
        *,
        experiment: Experiment,
        active_run: mlflow.ActiveRun,
        **kwargs: typing.Any,
    ) -> None:
        self.experiment = experiment
        self.active_run = active_run

    def log_metrics(
        self,
        *,
        epoch: int,
        mode: enums.Mode,
        n_processed_samples: int,
        values: dict[str, dto.NumberValue],
    ) -> None:
        for codename, value in values.items():
            mlflow.log_metric(
                key=f"{mode.value.lower()}/{codename}",
                value=float(value.value),
                step=n_processed_samples,
            )

    def close_logger(self) -> None:
        mlflow.end_run()
