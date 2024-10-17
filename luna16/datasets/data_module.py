import typing

import torch
from torch.utils import data as data_utils

from luna16.settings import settings
from luna16.utils import get_device

CandidateT = typing.TypeVar("CandidateT")


class DataModule(typing.Generic[CandidateT]):
    def __init__(
        self,
        batch_size: int,
        train: data_utils.Dataset[CandidateT],
        validation: data_utils.Dataset[CandidateT],
    ) -> None:
        self.batch_size = batch_size

        self.device, self.n_gpu_devices = get_device()
        self.is_using_cuda = self.device == torch.device("cuda")

        self.training_len = len(train)  # type: ignore
        self.validation_len = len(validation)  # type: ignore
        self.train_dl = self._create_training_dataloader(train)
        self.validation_dl = self._create_validation_dataloader(validation)

    def get_dataloader(self, train: bool) -> data_utils.DataLoader[CandidateT]:
        if train:
            return self.get_training_dataloader()
        else:
            return self.get_validation_dataloader()

    def get_training_dataloader(self) -> data_utils.DataLoader[CandidateT]:
        return self.train_dl

    def get_validation_dataloader(self) -> data_utils.DataLoader[CandidateT]:
        return self.validation_dl

    def _create_training_dataloader(
        self, train: data_utils.Dataset[CandidateT]
    ) -> data_utils.DataLoader[CandidateT]:
        batch_size = self.batch_size * self.n_gpu_devices
        train_dataloader = data_utils.DataLoader(
            dataset=train,
            batch_size=batch_size,
            num_workers=settings.NUM_WORKERS,
            pin_memory=self.is_using_cuda,
        )
        return train_dataloader

    def _create_validation_dataloader(
        self, validation: data_utils.Dataset[CandidateT]
    ) -> data_utils.DataLoader[CandidateT]:
        batch_size = self.batch_size * self.n_gpu_devices
        validation_dataloader = data_utils.DataLoader(
            dataset=validation,
            batch_size=batch_size,
            num_workers=settings.NUM_WORKERS,
            pin_memory=self.is_using_cuda,
        )
        return validation_dataloader
