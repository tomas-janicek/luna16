import datetime
import logging
import time
import typing

import mlflow

from luna16 import dto, services
from luna16.hyperparameters_container import HyperparameterContainer

from .. import messages
from .. import utils as message_utils

_log = logging.getLogger(__name__)


T = typing.TypeVar("T")


def log_metrics_to_console(
    message: messages.LogMetrics["dto.NumberMetric"],
    registry: "services.ServiceContainer",
) -> None:
    formatted_values = ", ".join(
        (
            f"{value.name.capitalize()}: {value.formatted_value}"
            for _, value in message.values.items()
        )
    )
    msg = f"E {message.epoch:04d} {message.mode.value:>10} " + formatted_values
    _log.info(msg)


def log_start_to_console(
    message: messages.LogStart,
    registry: "services.ServiceContainer",
) -> None:
    _log.info(f"Starting {message.training_description}")


def log_epoch_to_console(
    message: messages.LogEpoch,
    registry: "services.ServiceContainer",
) -> None:
    _log.info(
        f"E {message.epoch:04d} of {message.n_epochs:04d}, {message.training_length}/{message.validation_length} batches of "
        f"size {message.batch_size}"
    )


# `log_batch_to_console` is disabled because it was replaced
# with graphical `tqdm` progress bar.
def log_batch_to_console(
    message: messages.LogBatch,
    registry: "services.ServiceContainer",
) -> None:
    estimated_duration_in_seconds = (
        (time.time() - message.started_at)
        / (message.batch_index + 1)
        * (message.batch_size)
    )

    estimated_done_at = datetime.datetime.fromtimestamp(
        message.started_at + estimated_duration_in_seconds
    )
    estimated_duration = datetime.timedelta(seconds=estimated_duration_in_seconds)
    estimated_done_at_str = str(estimated_done_at).rsplit(".", 1)[0]
    estimated_duration_str = str(estimated_duration).rsplit(".", 1)[0]
    _log.info(
        f"E {message.epoch:04d} {message.mode.value:>10} {message.batch_index:-4}/{message.batch_size}, "
        f"done at {estimated_done_at_str}, {estimated_duration_str}"
    )


def log_batch_start_to_console(
    message: messages.LogBatchStart,
    registry: "services.ServiceContainer",
) -> None:
    _log.info(
        f"E {message.epoch:04d} {message.mode.value:>10} ----/{message.batch_size}, starting",
    )


def log_batch_end_to_console(
    message: messages.LogBatchEnd, registry: "services.ServiceContainer"
) -> None:
    now_dt = str(datetime.datetime.now()).rsplit(".", 1)[0]
    _log.info(
        f"E {message.epoch:04d} {message.mode.value:>10} ----/{message.batch_size}, done at {now_dt}"
    )


def log_metrics_to_tensorboard(
    message: messages.LogMetrics["dto.NumberMetric"],
    registry: "services.ServiceContainer",
) -> None:
    tensorboard_writer = message_utils.get_tensortboard_writer(
        mode=message.mode, registry=registry
    )
    for key, value in message.values.items():
        tensorboard_writer.add_scalar(
            tag=key,
            scalar_value=value.value,
            global_step=message.n_processed_samples,
        )

    tensorboard_writer.flush()


def log_results_to_tensorboard(
    message: messages.LogResult,
    registry: "services.ServiceContainer",
) -> None:
    tensorboard_writer = message_utils.get_tensortboard_writer(
        mode=message.mode, registry=registry
    )

    negative_label_mask = message.labels == 0
    positive_label_mask = message.labels == 1

    # Precision-Recall Curves
    tensorboard_writer.add_pr_curve(
        tag="pr",
        labels=message.labels,
        predictions=message.predictions,
        global_step=message.n_processed_samples,
    )

    negative_histogram_mask = negative_label_mask & (message.predictions > 0.01)
    positive_histogram_mask = positive_label_mask & (message.predictions < 0.99)

    bins = [x / 50.0 for x in range(51)]
    if negative_histogram_mask.any():
        tensorboard_writer.add_histogram(
            tag="is_neg",
            values=message.predictions[negative_histogram_mask],
            global_step=message.n_processed_samples,
            bins=bins,  # type: ignore
        )
    if positive_histogram_mask.any():
        tensorboard_writer.add_histogram(
            tag="is_pos",
            values=message.predictions[positive_histogram_mask],
            global_step=message.n_processed_samples,
            bins=bins,  # type: ignore
        )


def log_metrics_to_mlflow(
    message: messages.LogMetrics["dto.NumberMetric"],
    registry: "services.ServiceContainer",
) -> None:
    for codename, value in message.values.items():
        mlflow.log_metric(
            key=f"{message.mode.value.lower()}/{codename}",
            value=float(value.value),
            step=message.n_processed_samples,
        )


def log_model_to_mlflow(
    message: messages.LogModel,
    registry: "services.ServiceContainer",
) -> None:
    model_saver = registry.get_service(services.MLFlowModelSaver)
    model_saver.save_model(
        name=message.training_name, module=message.module, signature=message.signature
    )


def log_params_to_mlflow(
    message: messages.LogParams,
    registry: "services.ServiceContainer",
) -> None:
    hyperparameters = registry.get_service(HyperparameterContainer)
    params = hyperparameters.get_hyperparameters()
    mlflow.log_params(params)


def save_model(
    message: messages.LogModel,
    registry: "services.ServiceContainer",
) -> None:
    model_saver = registry.get_service(services.ModelSaver)
    model_saver.save_model(
        name=message.training_name, module=message.module, version=message.version
    )
