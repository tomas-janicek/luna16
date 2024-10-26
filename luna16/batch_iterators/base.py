import typing

from torch.utils import data as data_utils

from luna16 import enums

Key = typing.TypeVar("Key")
Value = typing.TypeVar("Value")


T = typing.TypeVar("T")


class BaseIteratorProvider:
    def enumerate_batches(
        self,
        enumerable: data_utils.DataLoader[typing.Any],
        *,
        epoch: int,
        mode: enums.Mode,
        candidate_batch_type: type[T],
    ) -> typing.Iterator[tuple[int, T]]: ...
