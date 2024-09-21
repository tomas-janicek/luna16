import typing
from dataclasses import dataclass

import torch

from luna16 import enums, services

T = typing.TypeVar("T")


@dataclass
class LogMessage:
    pass


class LogMetrics(LogMessage, typing.Generic[T]):
    epoch: int
    mode: enums.Mode
    n_processed_samples: int
    values: dict[str, T]


class LogStart(LogMessage):
    training_description: str


class LogEpoch(LogMessage):
    epoch: int
    n_epochs: int
    batch_size: int
    training_length: int
    validation_length: int


class LogBatchStart(LogMessage):
    epoch: int
    mode: enums.Mode
    batch_size: int


class LogBatch(LogMessage):
    epoch: int
    mode: enums.Mode
    batch_size: int
    batch_index: int
    started_at: float


class LogBatchEnd(LogMessage):
    epoch: int
    mode: enums.Mode
    batch_size: int


class LogResult(LogMessage):
    epoch: int
    mode: enums.Mode
    n_processed_samples: int
    labels: torch.Tensor
    predictions: torch.Tensor


class LogMessageHandler(typing.Protocol):
    def __call__(
        self, message: typing.Any, registry: services.ServiceContainer
    ) -> None: ...


LogMessageHandlersConfig = typing.Mapping[
    type[LogMessage], typing.Sequence[LogMessageHandler]
]
