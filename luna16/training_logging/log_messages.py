import typing

import svcs
from pydantic import BaseModel

from luna16 import enums

T = typing.TypeVar("T")


class LogMessage(BaseModel):
    pass


class MetricsLogMessage(LogMessage, typing.Generic[T]):
    epoch: int
    mode: enums.Mode
    n_processed_samples: int
    values: dict[str, T]


class LogMessageHandler(typing.Protocol):
    def __call__(self, message: typing.Any, services: svcs.Container) -> None: ...


LogMessageHandlersConfig = typing.Mapping[
    type[LogMessage], typing.Sequence[LogMessageHandler]
]
