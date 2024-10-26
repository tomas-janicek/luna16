import typing

import torch
from mlflow.models import infer_signature
from mlflow.pytorch import ModelSignature
from torch import nn
from torch.utils import data as data_utils

from luna16 import dto, message_handler, models, services


class SimpleCandidate(typing.NamedTuple):
    x: int
    y: int


class FakeMessageHandler(message_handler.BaseMessageHandler):
    def __init__(self, registry: services.ServiceContainer) -> None:
        self.requested_messages: list[message_handler.Message] = []
        self.registry = registry

    def handle_message(self, message: message_handler.Message) -> None:
        self.requested_messages.append(message)


class FakeDataset(data_utils.Dataset[SimpleCandidate]):
    def __init__(self, n: int) -> None:
        self.numbers = list(range(n))
        self.numbers_reversed = [*self.numbers]
        self.numbers_reversed.reverse()

    def __len__(self) -> int:
        return len(self.numbers)

    def __getitem__(self, index: int) -> SimpleCandidate:
        return SimpleCandidate(self.numbers[index], self.numbers_reversed[index])


class FakeModel(models.BaseModel[SimpleCandidate]):
    def __init__(self, module: nn.Module) -> None:
        self.module = module
        self.requested_training_params = []

    def fit_epoch(
        self,
        epoch: int,
        train_dl: data_utils.DataLoader[SimpleCandidate],
        validation_dl: data_utils.DataLoader[SimpleCandidate],
    ) -> dto.Scores:
        self.requested_training_params.append((epoch, train_dl, validation_dl))
        return dto.Scores()

    def get_module(self) -> torch.nn.Module: ...

    def get_signature(
        self, train_dl: data_utils.DataLoader[SimpleCandidate]
    ) -> ModelSignature:
        signature = infer_signature(
            model_input=[1, 2],
            model_output=[3],
        )
        return signature


class FakeModule(nn.Module):
    def __init__(self, in_features: int = 2, out_features: int = 1) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features=in_features, out_features=out_features)

    def forward(self, input_batch: torch.Tensor) -> torch.Tensor:
        return self.linear(input_batch)
