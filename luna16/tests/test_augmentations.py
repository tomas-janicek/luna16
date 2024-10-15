import math
import typing

import torch

from luna16 import augmentations


class FakeRandomProvider(augmentations.RandomProvider):
    def random_scalar(*args: typing.Any) -> float:
        return 1.0

    def random_angle(*args: typing.Any) -> float:
        return math.pi / 2

    def random_tensor_like(*args: typing.Any, tensor: torch.Tensor) -> torch.Tensor:
        return torch.ones(tensor.shape[0])


def test_flip() -> None:
    transformation_matrix = torch.eye(n=4)
    flipper = augmentations.Flip()
    transformation_matrix = flipper.apply_transformation(
        transformation=transformation_matrix
    )
    assert transformation_matrix[0, 0] == -1
    assert transformation_matrix[2, 2] == -1
    assert transformation_matrix[3, 3] == 1


def test_offset() -> None:
    transformation_matrix = torch.eye(n=4)
    offsetter = augmentations.Offset(offset=0.5, random_provider=FakeRandomProvider())
    transformation_matrix = offsetter.apply_transformation(
        transformation=transformation_matrix
    )
    # Check offset does not touch anything else
    assert transformation_matrix[0, 0] == 1.0
    assert transformation_matrix[2, 1] == 0.0
    assert transformation_matrix[3, 2] == 0.0
    assert transformation_matrix[3, 3] == 1.0
    # Offset is correctly set
    assert transformation_matrix[0, 3] == 0.5
    assert transformation_matrix[1, 3] == 0.5
    assert transformation_matrix[2, 3] == 0.5


def test_scale() -> None:
    transformation_matrix = torch.eye(n=4)
    scaler = augmentations.Scale(scale=0.5, random_provider=FakeRandomProvider())
    transformation_matrix = scaler.apply_transformation(
        transformation=transformation_matrix
    )
    # Check offset does not touch anything else
    assert transformation_matrix[0, 0] == 1.5
    assert transformation_matrix[1, 1] == 1.5
    assert transformation_matrix[2, 2] == 1.5
    # Offset is correctly set
    assert transformation_matrix[1, 3] == 0.0
    assert transformation_matrix[3, 3] == 1.0
    assert transformation_matrix[2, 1] == 0.0
    assert transformation_matrix[3, 2] == 0.0


def test_rotate() -> None:
    transformation_matrix = torch.eye(n=4)
    scaler = augmentations.Rotate(random_provider=FakeRandomProvider())
    transformation_matrix = scaler.apply_transformation(
        transformation=transformation_matrix
    )
    # Check offset does not touch anything else
    assert round(float(transformation_matrix[0, 0]), 1) == 0.0
    assert round(float(transformation_matrix[0, 1]), 1) == -1.0
    assert round(float(transformation_matrix[1, 0]), 1) == 1.0
    assert round(float(transformation_matrix[1, 1]), 1) == 0.0
    # Offset is correctly set
    assert transformation_matrix[1, 3] == 0.0
    assert transformation_matrix[2, 2] == 1.0
    assert transformation_matrix[3, 3] == 1.0
    assert transformation_matrix[2, 1] == 0.0
    assert transformation_matrix[3, 2] == 0.0
