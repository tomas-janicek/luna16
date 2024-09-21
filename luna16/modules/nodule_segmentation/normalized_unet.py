import math

import torch
from torch import nn as nn

from luna16 import enums

from .unet import UNet


class UNetNormalized(nn.Module):
    def __init__(
        self,
        in_channels: int,
        n_classes: int,
        depth: int,
        wf: int,
        padding: bool,
        batch_norm: bool,
        up_mode: enums.UpMode,
    ) -> None:
        super().__init__()

        self.input_batchnorm = nn.BatchNorm2d(num_features=in_channels)
        self.unet = UNet(
            in_channels=in_channels,
            n_classes=n_classes,
            depth=depth,
            wf=wf,
            padding=padding,
            batch_norm=batch_norm,
            up_mode=up_mode,
        )
        self.final = nn.Sigmoid()

        self._init_weights()

    def _init_weights(self) -> None:
        init_set = {
            nn.Conv2d,
            nn.Conv3d,
            nn.ConvTranspose2d,
            nn.ConvTranspose3d,
            nn.Linear,
        }
        for module in self.modules():
            if type(module) in init_set:
                # TODO: Explain why I used kaiming initialization
                nn.init.kaiming_normal_(
                    tensor=module.weight.data, mode="fan_out", nonlinearity="relu", a=0
                )
                if module.bias is not None:
                    _fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(  # type: ignore
                        module.weight.data
                    )
                    bound = 1 / math.sqrt(fan_out)
                    nn.init.normal_(module.bias, -bound, bound)

        # TODO: Use this or remove this!
        # nn.init.constant_(self.unet.last.bias, -4)
        # nn.init.constant_(self.unet.last.bias, 4)

    def forward(self, input_batch: torch.Tensor) -> torch.Tensor:
        bn_output: torch.Tensor = self.input_batchnorm(input_batch)
        un_output: torch.Tensor = self.unet(bn_output)
        fn_output: torch.Tensor = self.final(un_output)
        return fn_output
