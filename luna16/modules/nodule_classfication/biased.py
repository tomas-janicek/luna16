import typing

import torch
from torch import nn


class BiasedBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        conv_channels: int,
        **kwargs: typing.Any,
    ) -> None:
        super().__init__()

        self.conv1 = nn.Conv3d(
            in_channels=in_channels,
            out_channels=conv_channels,
            kernel_size=3,
            padding=1,
            bias=True,
        )
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv3d(
            in_channels=conv_channels,
            out_channels=conv_channels,
            kernel_size=3,
            padding=1,
            bias=True,
        )
        self.relu2 = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool3d(kernel_size=2, stride=2)

        nn.init.kaiming_normal_(self.conv1.weight, mode="fan_in", nonlinearity="relu")
        nn.init.zeros_(self.conv1.bias)  # type: ignore
        nn.init.kaiming_normal_(self.conv2.weight, mode="fan_in", nonlinearity="relu")
        nn.init.zeros_(self.conv2.bias)  # type: ignore

    def forward(self, input_batch: torch.Tensor) -> torch.Tensor:
        block_out = self.conv1(input_batch)
        block_out = self.relu1(block_out)
        block_out = self.conv2(block_out)
        block_out = self.relu2(block_out)

        return self.maxpool(block_out)
