import math
import random
import typing

import torch

TransformationMatrix = torch.Tensor


def random_scalar(*args: typing.Any) -> float:
    """Generates float number between -1 and 1

    Returns:
        float: number in range [-1, 1)
    """
    return random.random() * 2 - 1


def random_tensor_like(*args: typing.Any, tensor: torch.Tensor) -> torch.Tensor:
    return torch.rand_like(tensor)


def random_angle(*args: typing.Any) -> float:
    return random.random() * math.pi * 2


class Transformation(typing.Protocol):
    def apply_transformation(
        self, *, transformation: TransformationMatrix, **kwargs: typing.Any
    ) -> TransformationMatrix: ...


class Filter(typing.Protocol):
    def apply_filter(
        self, *, image: torch.Tensor, **kwargs: typing.Any
    ) -> torch.Tensor: ...


class Flip(Transformation):
    dimensions = 3

    def apply_transformation(
        self, *, transformation: torch.Tensor, **kwargs: typing.Any
    ) -> torch.Tensor:
        for i in range(self.dimensions):
            transformation[i, i] *= -1
        return transformation


class Offset(Transformation):
    random_provider = random_scalar
    dimensions = 3

    def __init__(self, offset: float) -> None:
        self.offset = offset

    def apply_transformation(
        self, *, transformation: torch.Tensor, **kwargs: typing.Any
    ) -> torch.Tensor:
        for i in range(self.dimensions):
            random_float = self.random_provider()
            transformation[i, 3] = self.offset * random_float
        return transformation


class Scale(Transformation):
    random_provider = random_scalar
    dimensions = 3

    def __init__(self, scale: float) -> None:
        self.scale = scale

    def apply_transformation(
        self, *, transformation: torch.Tensor, **kwargs: typing.Any
    ) -> torch.Tensor:
        for i in range(self.dimensions):
            random_float = self.random_provider()
            transformation[i, i] *= 1.0 + self.scale * random_float
        return transformation


class Rotate(Transformation):
    random_provider = random_angle

    def apply_transformation(
        self, *, transformation: torch.Tensor, **kwargs: typing.Any
    ) -> torch.Tensor:
        angle_radian = self.random_provider()
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
    random_provider = random_tensor_like

    def __init__(self, noise: float) -> None:
        self.noise = noise

    def apply_filter(
        self, *, image: torch.Tensor, **kwargs: typing.Any
    ) -> torch.Tensor:
        noise = self.random_provider(tensor=image)
        noise *= self.noise * noise
        image += noise
        return image
