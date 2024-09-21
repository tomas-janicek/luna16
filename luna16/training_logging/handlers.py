import logging
import typing

import svcs

from luna16.training_logging import log_messages

from . import dto

_log = logging.getLogger(__name__)


T = typing.TypeVar("T")


def log_metrics_to_console(
    message: log_messages.MetricsLogMessage[dto.NumberValue],
    services: svcs.Container,
) -> None:
    formatted_values = ", ".join(
        (
            f"{value.name.capitalize()}: {value.formatted_value}"
            for _, value in message.values.items()
        )
    )
    msg = f"E {message.epoch:04d} {message.mode.value:>10} " + formatted_values
    _log.info(msg)


LOG_MESSAGE_HANDLERS: log_messages.LogMessageHandlersConfig = {
    log_messages.LogMessage: (log_metrics_to_console,)
}
