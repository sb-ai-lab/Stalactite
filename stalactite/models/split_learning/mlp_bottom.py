import copy
import logging
import math
from functools import partial
from typing import Callable, List, Optional, Sequence, Union, Tuple, Any


import torch
from torch import nn, Tensor
from torchsummary import summary

from torchvision.models.efficientnet import MBConvConfig, FusedMBConvConfig, _MBConvConfig
from torchvision.utils import _log_api_usage_once
from torchvision.ops.misc import Conv2dNormActivation

logger = logging.getLogger(__name__)



class MLPBottom(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_channels: List[int],
        norm_layer: Optional[Callable[..., torch.nn.Module]] = None,
        activation_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.ReLU,
        bias: bool = True,
        dropout: float = 0.0,
        multilabel: bool = True,
        init_weights: float = None,
        class_weights: torch.Tensor = None
    ) -> None:

        super().__init__()
        _log_api_usage_once(self)

        if multilabel:
            self.criterion = torch.nn.BCEWithLogitsLoss(pos_weight=class_weights)
        else:
            raise ValueError()
            # self.criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

        layers = []
        in_dim = input_dim
        for hidden_dim in hidden_channels:
            layers.append(torch.nn.Linear(in_dim, hidden_dim, bias=bias))
            if norm_layer is not None:
                layers.append(norm_layer(hidden_dim))
            layers.append(activation_layer())
            layers.append(torch.nn.Dropout(dropout))
            in_dim = hidden_dim

        self.layers = nn.Sequential(*layers)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                if init_weights:
                    nn.init.constant_(m.weight, init_weights)
                else:
                    init_range = 1.0 / math.sqrt(m.out_features)
                    nn.init.uniform_(m.weight, -init_range, init_range)
                nn.init.zeros_(m.bias)

    def _forward_impl(self, x: Tensor) -> Tensor:
        x = self.layers(x)
        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)

    def update_weights(self, x: torch.Tensor, gradients: torch.Tensor, is_single: bool = False,
                       optimizer: torch.optim.Optimizer = None) -> None:

        optimizer.zero_grad()
        features = self.forward(x)
        features.backward(gradient=gradients)
        optimizer.step()

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)


if __name__ == "__main__":
    model = MLPBottom(input_dim=200, hidden_channels=[1000, 300, 100])
    summary(model, (200,), device="cpu", batch_size=3)

