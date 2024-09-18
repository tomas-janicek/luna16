import typing

import torch

from luna16 import enums

T = typing.TypeVar("T")


class BaseLoggingAdapter(typing.Generic[T], typing.Protocol):
    def log_all(
        self,
        *,
        epoch: int,
        mode: enums.Mode,
        n_processed_samples: int,
        values: dict[str, T],
    ) -> None: ...

    def close_all(self) -> None: ...


class BaseLoggerWrapper(typing.Protocol):
    def open_logger(self, **kwargs: typing.Any) -> None: ...

    def close_logger(self) -> None: ...


class TrainingProgressLoggerWrapper(BaseLoggerWrapper):
    def log_start(self) -> None: ...

    def log_epoch(self, *, epoch: int) -> None: ...


class BatchLoggerWrapper(BaseLoggerWrapper):
    def log_bach_start(
        self,
        *,
        epoch: int,
        mode: enums.Mode,
        batch_size: int,
    ) -> None: ...

    def log_batch(
        self,
        *,
        epoch: int,
        mode: enums.Mode,
        batch_size: int,
        batch_index: int,
        started_at: float,
    ) -> None: ...

    def log_bach_end(
        self,
        *,
        epoch: int,
        mode: enums.Mode,
        batch_size: int,
    ) -> None: ...


class MetricsLoggerWrapper(BaseLoggerWrapper, typing.Generic[T]):
    def log_metrics(
        self,
        *,
        epoch: int,
        mode: enums.Mode,
        n_processed_samples: int,
        values: dict[str, T],
    ) -> None: ...


class ResultLoggerWrapper(BaseLoggerWrapper):
    def log_result(
        self,
        *,
        epoch: int,
        mode: enums.Mode,
        n_processed_samples: int,
        labels: torch.Tensor,
        predictions: torch.Tensor,
    ) -> None: ...
