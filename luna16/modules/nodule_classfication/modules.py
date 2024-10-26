import math

import torch
from torch import nn


class LunaModel(nn.Module):
    def __init__(
        self, in_channels: int = 1, conv_channels: int = 8, out_features: int = 2
    ) -> None:
        super().__init__()

        # Tail
        self.tail_batchnorm = nn.BatchNorm3d(1)

        # Backbone
        self.block1 = LunaBlock(
            in_channels=in_channels,
            conv_channels=conv_channels,
        )
        self.block2 = LunaBlock(
            in_channels=conv_channels,
            conv_channels=conv_channels * 2,
        )
        self.block3 = LunaBlock(
            in_channels=conv_channels * 2,
            conv_channels=conv_channels * 4,
        )
        self.block4 = LunaBlock(
            in_channels=conv_channels * 4,
            conv_channels=conv_channels * 8,
        )

        # Head
        self.luna_head = LunaHead(in_features=1152, out_features=out_features)

        self._init_weights()

    # see also https://github.com/pytorch/pytorch/issues/18182
    def _init_weights(self):
        for m in self.modules():
            if type(m) in {
                nn.Linear,
                nn.Conv3d,
                nn.Conv2d,
                nn.ConvTranspose2d,
                nn.ConvTranspose3d,
            }:
                # TODO: Explain why I used kaiming initialization
                nn.init.kaiming_normal_(
                    tensor=m.weight.data,
                    a=0,
                    mode="fan_out",
                    nonlinearity="relu",
                )
                if m.bias is not None:
                    _fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(  # type: ignore
                        tensor=m.weight.data
                    )
                    bound = 1 / math.sqrt(fan_out)
                    nn.init.normal_(tensor=m.bias, mean=-bound, std=bound)

    def forward(self, input_batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        bn_output = self.tail_batchnorm(input_batch)

        block_out = self.block1(bn_output)
        block_out = self.block2(block_out)
        block_out = self.block3(block_out)
        block_out = self.block4(block_out)

        return self.luna_head(block_out)


class LunaBlock(nn.Module):
    def __init__(self, in_channels: int, conv_channels: int) -> None:
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

        # TODO: Add initialization

    def forward(self, input_batch: torch.Tensor) -> torch.Tensor:
        block_out = self.conv1(input_batch)
        block_out = self.relu1(block_out)
        block_out = self.conv2(block_out)
        block_out = self.relu2(block_out)

        return self.maxpool(block_out)


class LunaHead(nn.Module):
    def __init__(self, in_features: int = 1152, out_features: int = 2) -> None:
        super().__init__()

        self.head_linear = nn.Linear(in_features=in_features, out_features=out_features)
        self.head_softmax = nn.Softmax(dim=1)

        # TODO: Add initialization

    def forward(self, input_batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        conv_flat = input_batch.view(
            input_batch.size(0),
            -1,
        )
        linear_output = self.head_linear(conv_flat)

        return linear_output, self.head_softmax(linear_output)
