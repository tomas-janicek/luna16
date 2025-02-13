import math
import random
import typing

import torch

TransformationMatrix = torch.Tensor


class RandomProvider(typing.Protocol):
    def random_scalar(*args: typing.Any) -> float:
        """Generates float number between -1 and 1

        Returns:
            float: number in range [-1, 1)
        """
        ...

    def random_tensor_like(*args: typing.Any, tensor: torch.Tensor) -> torch.Tensor: ...

    def random_angle(*args: typing.Any) -> float: ...


class PythonRandomProvider(RandomProvider):
    def random_scalar(*args: typing.Any) -> float:
        return random.random() * 2 - 1

    def random_tensor_like(*args: typing.Any, tensor: torch.Tensor) -> torch.Tensor:
        return torch.rand_like(tensor)

    def random_angle(*args: typing.Any) -> float:
        return random.random() * math.pi * 2


class Transformation(typing.Protocol):
    random_provider: RandomProvider

    def apply_transformation(
        self, *, transformation: TransformationMatrix, **kwargs: typing.Any
    ) -> TransformationMatrix: ...


class Filter(typing.Protocol):
    random_provider: RandomProvider

    def apply_filter(
        self, *, image: torch.Tensor, **kwargs: typing.Any
    ) -> torch.Tensor: ...


class Flip(Transformation):
    def __init__(
        self, dimensions: int = 3, random_provider: RandomProvider | None = None
    ) -> None:
        self.dimensions = dimensions
        self.random_provider = random_provider or PythonRandomProvider()

    def apply_transformation(
        self, *, transformation: torch.Tensor, **kwargs: typing.Any
    ) -> torch.Tensor:
        for i in range(self.dimensions):
            transformation[i, i] *= -1
        return transformation


class Offset(Transformation):
    def __init__(
        self,
        offset: float,
        dimensions: int = 3,
        random_provider: RandomProvider | None = None,
    ) -> None:
        self.dimensions = dimensions
        self.random_provider = random_provider or PythonRandomProvider()
        self.offset = offset

    def apply_transformation(
        self, *, transformation: torch.Tensor, **kwargs: typing.Any
    ) -> torch.Tensor:
        for i in range(self.dimensions):
            random_float = self.random_provider.random_scalar()
            transformation[i, 3] = self.offset * random_float
        return transformation


class Scale(Transformation):
    def __init__(
        self,
        scale: float,
        dimensions: int = 3,
        random_provider: RandomProvider | None = None,
    ) -> None:
        self.scale = scale
        self.dimensions = dimensions
        self.random_provider = random_provider or PythonRandomProvider()

    def apply_transformation(
        self, *, transformation: torch.Tensor, **kwargs: typing.Any
    ) -> torch.Tensor:
        for i in range(self.dimensions):
            random_float = self.random_provider.random_scalar()
            transformation[i, i] *= 1.0 + self.scale * random_float
        return transformation


class Rotate(Transformation):
    def __init__(self, random_provider: RandomProvider | None = None) -> None:
        self.random_provider = random_provider or PythonRandomProvider()

    def apply_transformation(
        self, *, transformation: torch.Tensor, **kwargs: typing.Any
    ) -> torch.Tensor:
        angle_radian = self.random_provider.random_angle()
        sine = math.sin(angle_radian)
        cosine = math.cos(angle_radian)

        rotation_matrix = torch.tensor(
            [
                [cosine, -sine, 0, 0],
                [sine, cosine, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        )

        transformation @= rotation_matrix
        return transformation


class Noise(Filter):
    def __init__(
        self, noise: float, random_provider: RandomProvider | None = None
    ) -> None:
        self.noise = noise
        self.random_provider = random_provider or PythonRandomProvider()

    def apply_filter(
        self, *, image: torch.Tensor, **kwargs: typing.Any
    ) -> torch.Tensor:
        noise = self.random_provider.random_tensor_like(tensor=image)
        noise *= self.noise * noise
        image += noise
        return image
