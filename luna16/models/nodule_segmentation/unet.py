# MIT License
#
# Copyright (c) 2018 Joris
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Adapted from https://discuss.pytorch.org/t/unet-implementation/426

import torch
import torch.nn.functional as F
from torch import nn

from ... import enums


class UNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        n_classes: int = 2,
        depth: int = 5,
        wf: int = 6,
        padding: bool = False,
        batch_norm: bool = False,
        up_mode: enums.UpMode = enums.UpMode.UP_CONV,
    ) -> None:
        """
        Implementation of
        U-Net: Convolutional Networks for Biomedical Image Segmentation
        (Ronneberger et al., 2015)
        https://arxiv.org/abs/1505.04597

        Using the default arguments will yield the exact version used
        in the original paper

        Args:
            in_channels (int): number of input channels
            n_classes (int): number of output channels
            depth (int): depth of the network
            wf (int): number of filters in the first layer is 2**wf
                      in the second it is 2**wf + 1 and so on
            padding (bool): if True, apply padding such that the input shape
                            is the same as the output.
                            This may introduce artifacts
            batch_norm (bool): Use BatchNorm after layers with an
                               activation function
            up_mode (str): one of 'upconv' or 'upsample'.
                           'upconv' will use transposed convolutions for
                           learned upsampling.
                           'upsample' will use bilinear upsampling.
        """
        super().__init__()
        assert up_mode in (enums.UpMode.UP_CONV, enums.UpMode.UP_SAMPLE)
        self.padding = padding
        self.depth = depth
        prev_channels: int = in_channels
        self.down_path = nn.ModuleList()
        for i in range(depth):
            self.down_path.append(
                module=UNetConvBlock(
                    in_size=prev_channels,
                    out_size=2 ** (wf + i),
                    padding=padding,
                    batch_norm=batch_norm,
                )
            )
            prev_channels = 2 ** (wf + i)

        self.up_path = nn.ModuleList()
        for i in reversed(range(depth - 1)):
            self.up_path.append(
                module=UNetUpBlock(
                    in_size=prev_channels,
                    out_size=2 ** (wf + i),
                    up_mode=up_mode,
                    padding=padding,
                    batch_norm=batch_norm,
                )
            )
            prev_channels = 2 ** (wf + i)

        self.last = nn.Conv2d(
            in_channels=prev_channels, out_channels=n_classes, kernel_size=1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        blocks: list[torch.Tensor] = []
        for i, down in enumerate(self.down_path):
            x = down(x)
            if i != len(self.down_path) - 1:
                blocks.append(x)
                x = F.max_pool2d(input=x, kernel_size=2)

        for i, up in enumerate(self.up_path):
            x = up(x, blocks[-i - 1])

        return self.last(x)


class UNetConvBlock(nn.Module):
    def __init__(
        self, in_size: int, out_size: int, padding: bool, batch_norm: bool
    ) -> None:
        super().__init__()
        block = []

        block.append(
            nn.Conv2d(
                in_channels=in_size,
                out_channels=out_size,
                kernel_size=3,
                padding=int(padding),
            )
        )
        block.append(nn.ReLU())
        if batch_norm:
            block.append(nn.BatchNorm2d(num_features=out_size))

        block.append(
            nn.Conv2d(
                in_channels=out_size,
                out_channels=out_size,
                kernel_size=3,
                padding=int(padding),
            )
        )
        block.append(nn.ReLU())
        if batch_norm:
            block.append(nn.BatchNorm2d(num_features=out_size))

        self.block = nn.Sequential(*block)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.block(x)
        return out


class UNetUpBlock(nn.Module):
    def __init__(
        self,
        in_size: int,
        out_size: int,
        up_mode: enums.UpMode,
        padding: bool,
        batch_norm: bool,
    ) -> None:
        super().__init__()
        if up_mode == enums.UpMode.UP_CONV:
            self.up = nn.ConvTranspose2d(
                in_channels=in_size, out_channels=out_size, kernel_size=2, stride=2
            )
        elif up_mode == enums.UpMode.UP_SAMPLE:
            self.up = nn.Sequential(
                nn.Upsample(mode="bilinear", scale_factor=2),
                nn.Conv2d(in_channels=in_size, out_channels=out_size, kernel_size=1),
            )

        self.conv_block = UNetConvBlock(
            in_size=in_size, out_size=out_size, padding=padding, batch_norm=batch_norm
        )

    def forward(self, x: torch.Tensor, bridge: torch.Tensor) -> torch.Tensor:
        up: torch.Tensor = self.up(x)
        crop = self.center_crop(layer=bridge, target_size=up.shape[2:])
        out: torch.Tensor = torch.cat(tensors=[up, crop], dim=1)
        out = self.conv_block(out)

        return out

    def center_crop(self, layer: torch.Tensor, target_size: torch.Size) -> torch.Tensor:
        _, _, layer_height, layer_width = layer.size()
        diff_y = (layer_height - target_size[0]) // 2
        diff_x = (layer_width - target_size[1]) // 2
        return layer[
            :, :, diff_y : (diff_y + target_size[0]), diff_x : (diff_x + target_size[1])
        ]
