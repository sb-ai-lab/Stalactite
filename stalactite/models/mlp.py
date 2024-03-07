import logging
import math
from typing import Callable, List, Optional

import torch
from torch import nn, Tensor

from torchvision.utils import _log_api_usage_once

logger = logging.getLogger(__name__)


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_channels: List[int],
        norm_layer: Optional[Callable[..., torch.nn.Module]] = None,
        activation_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.ReLU,
        bias: bool = True,
        dropout: float = 0.0,
        init_weights: float = None,
    ) -> None:

        super().__init__()
        _log_api_usage_once(self)

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
        self.classifier = nn.Linear(in_dim, output_dim, bias=bias)

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
        x = self.classifier(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)

    def update_weights(self, x: torch.Tensor, y: torch.Tensor, is_single: bool = False,
                       optimizer: torch.optim.Optimizer = None, criterion: torch.nn.Module = None) -> None:
        optimizer.zero_grad()
        logit = self.forward(x)
        targets_type = torch.LongTensor if isinstance(criterion,
                                                      torch.nn.CrossEntropyLoss) else torch.FloatTensor
        loss = criterion(torch.squeeze(logit), y.type(targets_type))
        logger.info(f"loss: {loss.item()}")
        loss.backward()
        optimizer.step()

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)
