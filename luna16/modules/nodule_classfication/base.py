import typing

import pydantic
import torch
from torch import nn


class CnnParameters(pydantic.BaseModel):
    in_channels: int
    conv_channels: int
    out_features: int
    n_blocks: int
    input_dim: tuple[int, int, int]


class CnnModel(nn.Module):
    def __init__(
        self,
        name: str,
        in_channels: int,
        conv_channels: int,
        out_features: int,
        n_blocks: int,
        input_dim: tuple[int, int, int],
        block_class: type[nn.Module],
        **kwargs: typing.Any,
    ) -> None:
        super().__init__()
        self.name = name

        # Tail
        self.tail_batchnorm = nn.BatchNorm3d(in_channels)

        # Backbone
        self.luna_blocks = nn.ModuleList(
            [
                block_class(
                    in_channels=in_channels,
                    conv_channels=conv_channels,
                    **kwargs,
                )
            ]
        )
        out_conv_channels = conv_channels
        for _ in range(1, n_blocks):
            block = block_class(
                in_channels=out_conv_channels,
                conv_channels=out_conv_channels * 2,
                **kwargs,
            )
            self.luna_blocks.append(block)
            out_conv_channels *= 2

        # Calculate output features of network backbone.
        # For input dimension (32, 48, 48) and 4 blocks, the output dimensions are (1, 3, 3).
        output_dimensions: tuple[int, int, int] = (
            input_dim[0] // 2**n_blocks,
            input_dim[1] // 2**n_blocks,
            input_dim[2] // 2**n_blocks,
        )

        # Calculate in_features dynamically
        # It takes output dimensions of the last block and multiplies it by the number of channels in the last block.
        # If conv_channels = 8, then out_conv_channels = 64
        block_output_len = (
            out_conv_channels
            * output_dimensions[0]
            * output_dimensions[1]
            * output_dimensions[2]
        )

        # Head
        self.luna_head = CnnHead(
            in_features=block_output_len, out_features=out_features
        )

    def forward(self, input_batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        block_out = self.tail_batchnorm(input_batch)

        for block in self.luna_blocks:
            block_out = block(block_out)

        return self.luna_head(block_out)


class CnnHead(nn.Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()

        self.head_linear = nn.Linear(in_features=in_features, out_features=out_features)
        self.head_softmax = nn.Softmax(dim=1)

        nn.init.kaiming_normal_(
            self.head_linear.weight, mode="fan_in", nonlinearity="relu"
        )
        nn.init.zeros_(self.head_linear.bias)

    def forward(self, input_batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        conv_flat = torch.flatten(input_batch, start_dim=1)
        linear_output = self.head_linear(conv_flat)

        return linear_output, self.head_softmax(linear_output)
