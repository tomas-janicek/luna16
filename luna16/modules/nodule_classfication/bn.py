import torch
from torch import nn


class BNBlock(nn.Module):
    def __init__(self, in_channels: int, conv_channels: int) -> None:
        super().__init__()

        self.conv1 = nn.Conv3d(
            in_channels=in_channels,
            out_channels=conv_channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm3d(conv_channels)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv3d(
            in_channels=conv_channels,
            out_channels=conv_channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm3d(conv_channels)
        self.relu2 = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool3d(kernel_size=2, stride=2)

        nn.init.kaiming_normal_(self.conv1.weight, mode="fan_in", nonlinearity="relu")
        nn.init.kaiming_normal_(self.conv2.weight, mode="fan_in", nonlinearity="relu")

    def forward(self, input_batch: torch.Tensor) -> torch.Tensor:
        block_out = self.conv1(input_batch)
        block_out = self.bn1(block_out)
        block_out = self.relu1(block_out)
        block_out = self.conv2(block_out)
        block_out = self.bn2(block_out)
        block_out = self.relu2(block_out)

        return self.maxpool(block_out)
