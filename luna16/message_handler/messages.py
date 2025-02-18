import typing
from dataclasses import dataclass

import torch
from mlflow.pytorch import ModelSignature
from torch import nn

from luna16 import enums

if typing.TYPE_CHECKING:
    from luna16 import services

T = typing.TypeVar("T")
CandidateT = typing.TypeVar("CandidateT")


@dataclass
class Message:
    pass


@dataclass
class LogMetrics(Message, typing.Generic[T]):
    epoch: int
    mode: enums.Mode
    n_processed_samples: int
    values: dict[str, T]


@dataclass
class LogStart(Message):
    training_description: str


@dataclass
class LogEpoch(Message):
    epoch: int
    n_epochs: int
    batch_size: int
    training_length: int
    validation_length: int


@dataclass
class LogBatchStart(Message):
    epoch: int
    mode: enums.Mode
    batch_size: int


@dataclass
class LogBatch(Message):
    epoch: int
    mode: enums.Mode
    batch_size: int
    batch_index: int
    started_at: float


@dataclass
class LogBatchEnd(Message):
    epoch: int
    mode: enums.Mode
    batch_size: int


@dataclass
class LogResult(Message):
    epoch: int
    mode: enums.Mode
    n_processed_samples: int
    labels: torch.Tensor
    predictions: torch.Tensor


@dataclass
class LogParams(Message):
    pass


@dataclass
class LogModel(Message):
    module: nn.Module
    signature: ModelSignature
    training_name: str
    version: str


class LogMessageHandler(typing.Protocol):
    def __call__(
        self, message: typing.Any, registry: "services.ServiceContainer"
    ) -> None: ...


MessageHandlersConfig = typing.Mapping[
    type[Message], typing.Sequence[LogMessageHandler]
]
