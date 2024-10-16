import logging
from typing import Callable, List, Optional

import torch
from torch import nn, Tensor

from torchvision.utils import _log_api_usage_once

from stalactite.utils import init_linear_np

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
        class_weights: torch.Tensor = None,
        seed: int = None,
    ) -> None:

        super().__init__()
        _log_api_usage_once(self)

        self.input_dim = input_dim
        self.bias = bias
        self.dropout = dropout
        self.multilabel = multilabel
        self.init_weights = init_weights
        self.seed = seed

        if multilabel:
            self.criterion = torch.nn.BCEWithLogitsLoss(pos_weight=class_weights)

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
                    init_linear_np(m, seed=self.seed)
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

    @property
    def init_params(self):
        return {
            'input_dim': self.input_dim,
            'bias': self.bias,
            'dropout': self.dropout,
            'multilabel': self.multilabel,
            'init_weights': self.init_weights,
            'seed': self.seed,
        }
