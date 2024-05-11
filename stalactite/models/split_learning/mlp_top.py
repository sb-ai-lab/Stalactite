import logging

import torch
from torch import nn, Tensor

from torchvision.utils import _log_api_usage_once

from stalactite.utils import init_linear_np

logger = logging.getLogger(__name__)


class MLPTop(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        bias: bool = True,
        multilabel: bool = True,
        init_weights: float = None,
        class_weights: torch.Tensor = None,
        seed: int = None,
    ) -> None:

        super().__init__()
        _log_api_usage_once(self)
        self.seed = seed
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
                    init_linear_np(m, seed=self.seed)
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
            model_output = self.forward(x)
            model_output.backward(gradient=gradients)
            optimizer.step()

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)



