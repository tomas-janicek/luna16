import torch
from torch import nn


class Dropout3DModel(nn.Module):
    def __init__(
        self,
        in_channels: int,
        conv_channels: int,
        out_features: int,
        n_blocks: int,
        input_dim: tuple[int, int, int],
        dropout_rate: float,
    ) -> None:
        super().__init__()

        # Tail
        self.tail_batchnorm = nn.BatchNorm3d(in_channels)

        # Backbone
        self.luna_blocks = nn.ModuleList(
            [
                LunaBlock(
                    in_channels=in_channels,
                    conv_channels=conv_channels,
                    dropout_rate=dropout_rate,
                )
            ]
        )
        out_conv_channels = conv_channels
        for _ in range(1, n_blocks):
            block = LunaBlock(
                in_channels=out_conv_channels,
                conv_channels=out_conv_channels * 2,
                dropout_rate=dropout_rate,
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
        self.luna_head = LunaHead(
            in_features=block_output_len, out_features=out_features
        )

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
                    tensor=m.weight.data,  # type: ignore
                    a=0,
                    mode="fan_out",
                    nonlinearity="relu",
                )
                if m.bias is not None:
                    nn.init.zeros_(m.bias)  # type: ignore

    def forward(self, input_batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        block_out = self.tail_batchnorm(input_batch)

        for block in self.luna_blocks:
            block_out = block(block_out)

        return self.luna_head(block_out)


class LunaBlock(nn.Module):
    def __init__(
        self, in_channels: int, conv_channels: int, dropout_rate: float
    ) -> None:
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
        self.dropout1 = nn.Dropout3d(p=dropout_rate)

        self.conv2 = nn.Conv3d(
            in_channels=conv_channels,
            out_channels=conv_channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm3d(conv_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.dropout2 = nn.Dropout3d(p=dropout_rate)

        self.maxpool = nn.MaxPool3d(kernel_size=2, stride=2)

        # TODO: Add initialization

    def forward(self, input_batch: torch.Tensor) -> torch.Tensor:
        block_out = self.conv1(input_batch)
        block_out = self.bn1(block_out)
        block_out = self.relu1(block_out)
        block_out = self.dropout1(block_out)
        block_out = self.conv2(block_out)
        block_out = self.bn2(block_out)
        block_out = self.relu2(block_out)
        block_out = self.dropout2(block_out)

        return self.maxpool(block_out)


class LunaHead(nn.Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()

        self.head_linear = nn.Linear(in_features=in_features, out_features=out_features)
        self.head_softmax = nn.Softmax(dim=1)

        # TODO: Add initialization

    def forward(self, input_batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        conv_flat = torch.flatten(input_batch, start_dim=1)
        linear_output = self.head_linear(conv_flat)

        return linear_output, self.head_softmax(linear_output)
