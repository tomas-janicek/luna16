import typing
from dataclasses import dataclass

import numpy as np
import pydantic
import torch
from numpy import typing as np_typing

from luna16 import enums


class CoordinatesXYZ(pydantic.BaseModel):
    x: float
    y: float
    z: float

    def __init__(self, x: int, y: int, z: int) -> None:
        super().__init__(x=x, y=y, z=z)
        self._array = np.array([self.x, self.y, self.z])

    def get_array(self) -> np_typing.NDArray[np.float32]:
        return self._array

    def to_irc(
        self,
        origin: "CoordinatesXYZ",
        voxel_size: "CoordinatesXYZ",
        transformation_direction: np_typing.NDArray[np.float32],
    ) -> "CoordinatesIRC":
        coords_CRI = (
            (self._array - origin.get_array()) @ np.linalg.inv(transformation_direction)
        ) / voxel_size.get_array()
        coords_CRI = np.round(coords_CRI)
        return CoordinatesIRC(
            index=int(coords_CRI[2]), row=int(coords_CRI[1]), col=int(coords_CRI[0])
        )

    def __getitem__(self, index: int) -> int:
        return self._array[index]

    def __len__(self) -> int:
        return len(self._array)

    def __iter__(self) -> typing.Iterator[int]:  # type: ignore
        return iter(self._array)


class CoordinatesIRC(pydantic.BaseModel):
    index: int
    row: int
    col: int

    def __init__(self, index: int, row: int, col: int) -> None:
        super().__init__(index=index, row=row, col=col)
        self._array = np.array([self.index, self.row, self.col])

    def get_array(self) -> np_typing.NDArray[np.float32]:
        return self._array

    def to_xyz(
        self,
        origin: "CoordinatesXYZ",
        voxel_size: "CoordinatesXYZ",
        transformation_direction: np_typing.NDArray[np.float32],
    ) -> "CoordinatesXYZ":
        x, y, z = (
            transformation_direction @ (self._array * voxel_size.get_array())
        ) + origin.get_array()
        return CoordinatesXYZ(x=x, y=y, z=z)

    def __getitem__(self, index: int) -> int:
        return self._array[index]

    def __len__(self) -> int:
        return len(self._array)

    def __iter__(self) -> typing.Iterator[int]:  # type: ignore
        return iter(self._array)

    def move_dimension(
        self, *, dimension: enums.DimensionIRC, move_by: int
    ) -> "CoordinatesIRC":
        match dimension:
            case enums.DimensionIRC.INDEX:
                self.index += move_by
            case enums.DimensionIRC.ROW:
                self.row += move_by
            case enums.DimensionIRC.COL:
                self.col += move_by
        return self


class CandidateInfo(pydantic.BaseModel):
    series_uid: str
    is_nodule: bool
    diameter_mm: float
    center: CoordinatesXYZ


class CandidateMalignancyInfo(pydantic.BaseModel):
    series_uid: str
    is_nodule: bool
    is_annotated: bool
    is_malignant: bool
    diameter_mm: float
    center: CoordinatesXYZ


class LunaClassificationCandidate(typing.NamedTuple):
    candidate: torch.Tensor
    labels: torch.Tensor
    series_uid: str
    center_irc: torch.Tensor


class LunaSegmentationCandidate(typing.NamedTuple):
    candidate: torch.Tensor
    positive_candidate_mask: torch.Tensor
    series_uid: str
    slice_index: int


class Ratio:
    def __init__(self, ratios: list[int]) -> None:
        self.ratios = ratios
        self.cycle: int = sum(self.ratios)
        self.intervals = []
        start = 0
        end = 0
        for ratio in self.ratios:
            end += ratio
            interval = range(start, end)
            start += ratio
            self.intervals.append(interval)

    def get_class(self, index: int) -> tuple[int, int]:
        pure_index = index % self.cycle
        for i, interval in enumerate(self.intervals):
            if pure_index in interval:
                n_cycle = index // self.cycle
                return i, n_cycle * len(interval)
        raise NotImplementedError(
            "Something went wrong during getting class from ratio."
        )


class LunaClassificationRatio(Ratio):
    def __init__(self, positive: int, negative: int) -> None:
        super().__init__([positive, negative])


class LunaMalignantRatio(Ratio):
    def __init__(self, malignant: int, benign: int, not_module: int) -> None:
        super().__init__([malignant, benign, not_module])


class SegmentationBatchMetrics:
    def __init__(
        self,
        loss: torch.Tensor,
        false_negative_loss: torch.Tensor,
        true_positive: torch.Tensor,
        false_negative: torch.Tensor,
        false_positive: torch.Tensor,
    ) -> None:
        self.loss = loss
        self.false_negative_loss = false_negative_loss
        self.true_positive = true_positive
        self.false_negative = false_negative
        self.false_positive = false_positive

    @classmethod
    def create_empty(
        cls, dataset_len: int, device: torch.device
    ) -> "SegmentationBatchMetrics":
        return cls(
            loss=torch.tensor([]).to(device),
            false_negative_loss=torch.tensor([]).to(device),
            true_positive=torch.tensor([]).to(device),
            false_negative=torch.tensor([]).to(device),
            false_positive=torch.tensor([]).to(device),
        )

    def add_batch_metrics(
        self,
        loss: torch.Tensor | None = None,
        false_negative_loss: torch.Tensor | None = None,
        hard_true_positive: torch.Tensor | None = None,
        hard_false_negative: torch.Tensor | None = None,
        hard_false_positive: torch.Tensor | None = None,
    ) -> None:
        if loss is not None:
            self.loss = torch.cat((self.loss, loss), dim=0)
        if false_negative_loss is not None:
            self.false_negative_loss = torch.cat(
                (self.false_negative_loss, false_negative_loss), dim=0
            )
        if hard_true_positive is not None:
            self.true_positive = torch.cat(
                (self.true_positive, hard_true_positive), dim=0
            )
        if hard_false_negative is not None:
            self.false_negative = torch.cat(
                (self.false_negative, hard_false_negative), dim=0
            )
        if hard_false_positive is not None:
            self.false_positive = torch.cat(
                (self.false_positive, hard_false_positive), dim=0
            )


class ClassificationBatchMetrics:
    def __init__(
        self,
        loss: torch.Tensor,
        labels: torch.Tensor,
        predictions: torch.Tensor,
    ) -> None:
        self.loss = loss
        self.labels = labels
        self.predictions = predictions

    @classmethod
    def create_empty(
        cls, dataset_len: int, device: torch.device
    ) -> "ClassificationBatchMetrics":
        return cls(
            loss=torch.tensor([]).to(device),
            labels=torch.tensor([]).to(device),
            predictions=torch.tensor([]).to(device),
        )

    def add_batch_metrics(
        self,
        loss: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        predictions: torch.Tensor | None = None,
    ) -> None:
        if loss is not None:
            self.loss = torch.cat((self.loss, loss), dim=0)
        if labels is not None:
            self.labels = torch.cat((self.labels, labels), dim=0)
        if predictions is not None:
            self.predictions = torch.cat((self.predictions, predictions), dim=0)

    def dataset_length(self) -> int:
        return self.labels.shape[0]


@dataclass
class Value:
    name: str
    value: typing.Any


@dataclass
class NumberValue(Value):
    name: str
    value: float | np.float32
    formatted_value: str


Scores = dict[str, float | np.float32]
