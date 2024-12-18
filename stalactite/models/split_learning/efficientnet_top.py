import logging
from functools import partial
from typing import Optional, Sequence, Union, Tuple

import torch
from torch import nn, Tensor

from torchvision.models.efficientnet import MBConvConfig, FusedMBConvConfig
from torchvision.utils import _log_api_usage_once

from stalactite.utils import init_linear_np

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


class EfficientNetTop(nn.Module):
    def __init__(
            self,
            dropout: float = 0.1,
            input_dim=None,
            num_classes: int = 1000,
            init_weights: float = None,
            seed: int = None,

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

        self.dropout = dropout
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.init_weights = init_weights

        self.criterion = torch.nn.CrossEntropyLoss()
        self.seed = seed
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(input_dim, num_classes),
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                if init_weights:
                    nn.init.constant_(m.weight, init_weights)
                else:
                    init_linear_np(m, seed=self.seed)
                nn.init.zeros_(m.bias)

    def _forward_impl(self, x: Tensor) -> Tensor:

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)

    def update_weights(self, x: torch.Tensor, gradients: torch.Tensor, is_single: bool = False,
                       optimizer: torch.optim.Optimizer = None) -> Optional[Tensor]:
        optimizer.zero_grad()
        if is_single:
            logit = self.forward(x)
            loss = self.criterion(torch.squeeze(logit), gradients.type(torch.LongTensor))
            logger.info(f"Loss: {loss.item()}")
            grads = torch.autograd.grad(outputs=loss, inputs=x, retain_graph=True)
            loss.backward()
            optimizer.step()
            return grads[0]
        else:
            model_output = self.forward(x)
            model_output.backward(gradient=gradients)
            optimizer.step()

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)

    def get_weights(self) -> torch.Tensor:
        return self.linear.weight.clone()

    @property
    def init_params(self):
        return {
            'dropout': self.dropout,
            'input_dim': self.input_dim,
            'num_classes': self.num_classes,
            'init_weights': self.init_weights,
            'seed': self.seed,
        }
