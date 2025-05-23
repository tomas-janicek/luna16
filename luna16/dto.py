import typing
from dataclasses import dataclass

import numpy as np
import pydantic
import torch
from numpy import floating
from numpy import typing as np_typing

from luna16 import enums

FloatType = float | np.floating


class CoordinatesXYZ(pydantic.BaseModel):
    x: float
    y: float
    z: float

    def __init__(self, x: int, y: int, z: int) -> None:
        super().__init__(x=x, y=y, z=z)
        self._array = np.array([self.x, self.y, self.z], dtype=np.float32)

    def __getitem__(self, index: int) -> int:
        return self._array[index]

    def __len__(self) -> int:
        return len(self._array)

    def __iter__(self) -> typing.Iterator[int]:  # type: ignore
        return iter(self._array)

    def __hash__(self) -> int:
        return hash((self.x, self.y, self.z))

    def __eq__(self, other: typing.Any) -> bool:
        if not isinstance(other, self.__class__):
            raise NotImplementedError
        return (self.x, self.y, self.z) == (other.x, other.y, other.z)

    def __repr__(self) -> str:
        _repr = f"{self.__class__.__name__}(x={self.x}, y={self.y}, z={self.z})"
        return _repr

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


class CoordinatesIRC(pydantic.BaseModel):
    index: int
    row: int
    col: int

    def __init__(self, index: int, row: int, col: int) -> None:
        super().__init__(index=index, row=row, col=col)
        self._array = np.array([self.index, self.row, self.col], dtype=np.int16)

    def __getitem__(self, index: int) -> int:
        return self._array[index]

    def __len__(self) -> int:
        return len(self._array)

    def __iter__(self) -> typing.Iterator[np.int16]:  # type: ignore
        return iter(self._array)

    def __hash__(self) -> int:
        return hash((self.index, self.row, self.col))

    def __eq__(self, other: typing.Any) -> bool:
        if not isinstance(other, self.__class__):
            raise NotImplementedError
        return (self.index, self.row, self.col) == (other.index, other.row, other.col)

    def __repr__(self) -> str:
        _repr = (
            f"{self.__class__.__name__}("
            f"index={self.index}, "
            f"row={self.row}, "
            f"col={self.col})"
        )
        return _repr

    def get_array(self) -> np_typing.NDArray[np.int16]:
        return self._array

    def to_xyz(
        self,
        origin: "CoordinatesXYZ",
        voxel_size: "CoordinatesXYZ",
        transformation_direction: np_typing.NDArray[np.int16],
    ) -> "CoordinatesXYZ":
        x, y, z = (
            transformation_direction @ (self._array * voxel_size.get_array())
        ) + origin.get_array()
        return CoordinatesXYZ(x=x, y=y, z=z)

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


class CandidateMetadata(typing.NamedTuple):
    series_uid: str
    candidate_class: enums.CandidateClass
    diameter_mm: float
    file_path: str

    def __hash__(self) -> int:
        return hash(self.file_path)


class LunaClassificationCandidate(typing.NamedTuple):
    candidate: torch.Tensor
    labels: torch.Tensor
    series_uid: str
    center_irc: torch.Tensor


class LunaClassificationCandidateBatch(typing.NamedTuple):
    candidate: torch.Tensor
    labels: torch.Tensor
    series_uid: torch.Tensor
    center_irc: torch.Tensor


class Ratio:
    def __init__(self, ratios: list[int]) -> None:
        self._validate_ratios(ratios)

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

    def _validate_ratios(self, ratios: list[int]) -> None:
        for ratio in ratios:
            if ratio < 0:
                raise ValueError("Ratio must be positive number or zero.")


class NoduleRatio(Ratio):
    def __init__(self, positive: int, negative: int) -> None:
        super().__init__([positive, negative])

    def __repr__(self) -> str:
        return f"Positive: {self.ratios[0]}, Negative: {self.ratios[1]}"


class MalignantRatio(Ratio):
    def __init__(self, malignant: int, benign: int, not_module: int) -> None:
        super().__init__([malignant, benign, not_module])

    def __repr__(self) -> str:
        return f"Malignant: {self.ratios[0]}, Benign: {self.ratios[1]}, Not module: {self.ratios[2]}"


@dataclass
class Metric:
    name: str
    value: typing.Any


@dataclass
class NumberMetric(Metric):
    name: str
    value: float | np.float32 | floating
    formatted_value: str


Scores = dict[str, float | np.float32]
