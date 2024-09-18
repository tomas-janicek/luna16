import math
import random

import torch
from torch import nn
from torch.nn import functional as F


class SegmentationAugmentation(nn.Module):
    def __init__(
        self,
        flip: bool | None = None,
        offset: float | None = None,
        scale: float | None = None,
        rotate: bool | None = None,
        noise: float | None = None,
    ) -> None:
        super().__init__()
        self.flip = flip
        self.offset = offset
        self.scale = scale
        self.rotate = rotate
        self.noise = noise

    def forward(
        self, input: torch.Tensor, labels: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # TODO: Debug what this actually do
        transformation_matrix = self.build_2d_transformation_matrix()
        transformation_matrix = transformation_matrix.expand(input.shape[0], -1, -1)
        transformation_matrix = transformation_matrix.to(input.device, torch.float32)

        affine_transformation = F.affine_grid(
            theta=transformation_matrix[:, :2],
            size=list(input.size()),
            align_corners=False,
        )

        augmented_input = F.grid_sample(
            input=input,
            grid=affine_transformation,
            # padding_mode="border",
            align_corners=False,
        )
        augmented_labels = F.grid_sample(
            input=labels.to(torch.float32),
            grid=affine_transformation,
            # padding_mode="border",
            align_corners=False,
        )

        if self.noise:
            noise = torch.randn_like(augmented_input) * self.noise
            augmented_input += noise

        return augmented_input, augmented_labels > 0.5

    def build_2d_transformation_matrix(self) -> torch.Tensor:
        transformation_matrix = torch.eye(3)

        for i in range(2):
            if self.flip:
                if random.random() > 0.5:
                    transformation_matrix[i, i] *= -1

            if self.offset:
                offset_float = self.offset
                random_float = random.random() * 2 - 1
                transformation_matrix[2, i] = offset_float * random_float

            if self.scale:
                scale_float = self.scale
                random_float = random.random() * 2 - 1
                transformation_matrix[i, i] *= 1.0 + scale_float * random_float

        if self.rotate:
            angle_radiant = random.random() * math.pi * 2
            s = math.sin(angle_radiant)
            c = math.cos(angle_radiant)
            rotation = torch.tensor(
                [
                    [c, -s, 0],
                    [s, c, 0],
                    [0, 0, 1],
                ]
            )
            transformation_matrix @= rotation

        return transformation_matrix

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"flip={self.flip}, "
            f"scale={self.scale}, "
            f"noise={self.noise}, "
            f"offset={self.offset}, "
            f"rotate={self.rotate})"
        )
