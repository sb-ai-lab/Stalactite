import copy
import logging
import math
from functools import partial
from typing import Callable, List, Optional, Sequence, Union, Tuple


import torch
from torch import nn, Tensor

from torchvision.models.efficientnet import MBConvConfig, FusedMBConvConfig
from torchvision.utils import _log_api_usage_once
from torchvision.ops.misc import Conv2dNormActivation

logger = logging.getLogger(__name__)

def _efficientnet_conf(
    width_mult: float,
    depth_mult: float
) -> Tuple[Sequence[Union[MBConvConfig, FusedMBConvConfig]], Optional[int]]:
    inverted_residual_setting: Sequence[Union[MBConvConfig, FusedMBConvConfig]]
    bneck_conf = partial(MBConvConfig, width_mult=width_mult, depth_mult=depth_mult)
    inverted_residual_setting = [
        bneck_conf(1, 3, 1, 32, 16, 1),
        bneck_conf(6, 3, 2, 16, 24, 2),
        bneck_conf(6, 5, 2, 24, 40, 2),
        bneck_conf(6, 3, 2, 40, 80, 3),
        bneck_conf(6, 5, 1, 80, 112, 3),
        bneck_conf(6, 5, 2, 112, 192, 4),
        bneck_conf(6, 3, 1, 192, 320, 1),
    ]
    last_channel = None

    return inverted_residual_setting, last_channel


class EfficientNet(nn.Module):
    def __init__(
        self,
        width_mult: float = 1.0,
        depth_mult: float = 1.0,
        dropout: float = 0.1,
        stochastic_depth_prob: float = 0.2,
        num_classes: int = 1000,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        init_weights: float = None
    ) -> None:
        """
        EfficientNet V1 and V2 main class

        Args:
            dropout (float): The droupout probability
            stochastic_depth_prob (float): The stochastic depth probability
            num_classes (int): Number of classes
            norm_layer (Optional[Callable[..., nn.Module]]): Module specifying the normalization layer to use
        """
        super().__init__()
        _log_api_usage_once(self)

        self.width_mult = width_mult
        self.depth_mult = depth_mult
        self.dropout = dropout
        self.stochastic_depth_prob = stochastic_depth_prob
        self.num_classes = num_classes

        self.criterion = torch.nn.CrossEntropyLoss()

        inverted_residual_setting, last_channel = _efficientnet_conf(width_mult=width_mult, depth_mult=depth_mult)

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        layers: List[nn.Module] = []

        # building first layer
        firstconv_output_channels = inverted_residual_setting[0].input_channels
        layers.append(
            Conv2dNormActivation(
                1, firstconv_output_channels, kernel_size=3, stride=2, norm_layer=norm_layer,
                activation_layer=nn.SiLU
            )
        )

        # building inverted residual blocks
        total_stage_blocks = sum(cnf.num_layers for cnf in inverted_residual_setting)
        stage_block_id = 0
        for cnf in inverted_residual_setting:
            stage: List[nn.Module] = []
            for _ in range(cnf.num_layers):
                # copy to avoid modifications. shallow copy is enough
                block_cnf = copy.copy(cnf)

                # overwrite info if not the first conv in the stage
                if stage:
                    block_cnf.input_channels = block_cnf.out_channels
                    block_cnf.stride = 1

                # adjust stochastic depth probability based on the depth of the stage block
                sd_prob = stochastic_depth_prob * float(stage_block_id) / total_stage_blocks

                stage.append(block_cnf.block(block_cnf, sd_prob, norm_layer))
                stage_block_id += 1

            layers.append(nn.Sequential(*stage))

        # building last several layers
        lastconv_input_channels = inverted_residual_setting[-1].out_channels
        lastconv_output_channels = last_channel if last_channel is not None else 4 * lastconv_input_channels
        layers.append(
            Conv2dNormActivation(
                lastconv_input_channels,
                lastconv_output_channels,
                kernel_size=1,
                norm_layer=norm_layer,
                activation_layer=nn.SiLU,
            )
        )

        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(lastconv_output_channels, num_classes),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if init_weights:
                    nn.init.constant_(m.weight, init_weights)
                else:
                    nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                if init_weights:
                    nn.init.constant_(m.weight, init_weights)
                else:
                    init_range = 1.0 / math.sqrt(m.out_features)
                    nn.init.uniform_(m.weight, -init_range, init_range)
                nn.init.zeros_(m.bias)

    def _forward_impl(self, x: Tensor) -> Tensor:
        x = self.features(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        x = self.classifier(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)

    def update_weights(self, x: torch.Tensor, y: torch.Tensor, is_single: bool = False,
                       optimizer: torch.optim.Optimizer = None) -> None:
        optimizer.zero_grad()
        logit = self.forward(x)
        loss = self.criterion(torch.squeeze(logit), y.type(torch.LongTensor))
        logger.info(f"Loss: {loss.item()}")
        loss.backward()
        optimizer.step()

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)

    def get_weights(self) -> torch.Tensor:
        return self.linear.weight.clone()

    @property
    def init_params(self):
        return {
            'width_mult': self.width_mult,
            'depth_mult': self.depth_mult,
            'dropout': self.dropout,
            'stochastic_depth_prob': self.stochastic_depth_prob,
            'num_classes': self.num_classes,
        }


