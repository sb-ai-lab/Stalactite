import logging
from typing import Optional

import torch

logger = logging.getLogger(__name__)


class LogisticRegressionBatch(torch.nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        init_weights: float = None,
    ):
        """
        Args:
            input_dim (int): input dimension
            output_dim (int): output dimension
        """
        super(LogisticRegressionBatch, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim, bias=False, device=None, dtype=None)
        if init_weights is not None:
            self.linear.weight.data = torch.full((output_dim, input_dim), init_weights)

    def forward(self, x: torch.Tensor):
        return self.linear(x)

    def update_weights(self, x: torch.Tensor, gradients: torch.Tensor, is_single: bool = False, criterion=None,
                       optimizer: torch.optim.Optimizer = None) -> None:
        optimizer.zero_grad()
        logit = self.forward(x)
        if is_single:
            targets_type = torch.LongTensor if isinstance(self._criterion,
                                                          torch.nn.CrossEntropyLoss) else torch.FloatTensor
            loss = criterion(torch.squeeze(logit), gradients.type(targets_type))
            loss.backward()
        else:
            logit.backward(gradient=gradients)
        optimizer.step()

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)

    def get_weights(self) -> torch.Tensor:
        return self.linear.weight.clone()
