import typing
from dataclasses import dataclass

import torch
from torch import nn
from torch.utils import data as data_utils

from luna16 import enums, services

T = typing.TypeVar("T")
CandidateT = typing.TypeVar("CandidateT")


@dataclass
class LogMessage:
    pass


@dataclass
class LogMetrics(LogMessage, typing.Generic[T]):
    epoch: int
    mode: enums.Mode
    n_processed_samples: int
    values: dict[str, T]


@dataclass
class LogStart(LogMessage):
    training_description: str


@dataclass
class LogEpoch(LogMessage):
    epoch: int
    n_epochs: int
    batch_size: int
    training_length: int
    validation_length: int


@dataclass
class LogBatchStart(LogMessage):
    epoch: int
    mode: enums.Mode
    batch_size: int


@dataclass
class LogBatch(LogMessage):
    epoch: int
    mode: enums.Mode
    batch_size: int
    batch_index: int
    started_at: float


@dataclass
class LogBatchEnd(LogMessage):
    epoch: int
    mode: enums.Mode
    batch_size: int


@dataclass
class LogResult(LogMessage):
    epoch: int
    mode: enums.Mode
    n_processed_samples: int
    labels: torch.Tensor
    predictions: torch.Tensor


@dataclass
class LogImages(LogMessage, typing.Generic[CandidateT]):
    epoch: int
    mode: enums.Mode
    n_processed_samples: int
    dataloader: data_utils.DataLoader[CandidateT]
    model: nn.Module
    device: torch.device


class LogMessageHandler(typing.Protocol):
    def __call__(
        self, message: typing.Any, registry: services.ServiceContainer
    ) -> None: ...


LogMessageHandlersConfig = typing.Mapping[
    type[LogMessage], typing.Sequence[LogMessageHandler]
]
