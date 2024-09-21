import typing

from luna16 import datasets, models

CandidateT = typing.TypeVar("CandidateT")


class BaseTrainer(typing.Protocol[CandidateT]):
    def fit(
        self,
        *,
        model: models.BaseModel[CandidateT],
        epochs: int,
        data_module: datasets.DataModule[CandidateT],
    ) -> None: ...

    def fit_epoch(
        self,
        *,
        model: models.BaseModel[CandidateT],
        epoch: int,
        data_module: datasets.DataModule[CandidateT],
    ) -> None: ...
