import typing

import torch
from mlflow.pytorch import ModelSignature
from torch.utils import data as data_utils

from luna16 import scoring

CandidateT = typing.TypeVar("CandidateT")


class BaseModel(typing.Protocol[CandidateT]):  # type: ignore
    def fit_epoch(
        self,
        epoch: int,
        train_dl: data_utils.DataLoader[CandidateT],
        validation_dl: data_utils.DataLoader[CandidateT],
    ) -> scoring.PerformanceMetrics: ...

    def get_module(self) -> torch.nn.Module: ...

    def get_signature(
        self, train_dl: data_utils.DataLoader[CandidateT]
    ) -> ModelSignature: ...
