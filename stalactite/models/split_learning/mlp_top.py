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



class MLPTop(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        bias: bool = True,
        multilabel: bool = True,
        init_weights: float = None,
        class_weights: torch.Tensor = None
    ) -> None:

        super().__init__()
        _log_api_usage_once(self)

        if multilabel:
            self.criterion = torch.nn.BCEWithLogitsLoss(pos_weight=class_weights)
        else:
            self.criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

        self.classifier = nn.Linear(input_dim, output_dim, bias=bias)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                if init_weights:
                    nn.init.constant_(m.weight, init_weights)
                else:
                    init_range = 1.0 / math.sqrt(m.out_features)
                    nn.init.uniform_(m.weight, -init_range, init_range)
                nn.init.zeros_(m.bias)

    def _forward_impl(self, x: Tensor) -> Tensor:
        x = self.classifier(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)

    def update_weights(self, x: torch.Tensor, gradients: torch.Tensor, is_single: bool = False,
                       optimizer: torch.optim.Optimizer = None) -> Tensor:
        optimizer.zero_grad()

        if is_single:
            logit = self.forward(x)
            loss = self.criterion(torch.squeeze(logit), gradients.type(torch.FloatTensor))
            grads = torch.autograd.grad(outputs=loss, inputs=x, retain_graph=True)
            loss.backward()
            optimizer.step()
            return grads[0]
        else:
            x.backward(gradient=gradients)
            optimizer.step()

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)



