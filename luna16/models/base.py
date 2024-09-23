import typing

import torch
from torch.utils import data as data_utils

CandidateT = typing.TypeVar("CandidateT")


class BaseModel(typing.Protocol[CandidateT]):  # type: ignore
    def fit_epoch(
        self,
        epoch: int,
        train_dl: data_utils.DataLoader[CandidateT],
        validation_dl: data_utils.DataLoader[CandidateT],
    ) -> None: ...

    def get_module(self) -> torch.nn.Module: ...
