import logging
from typing import Optional

import torch

logger = logging.getLogger(__name__)


class LogisticRegressionBatch(torch.nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        learning_rate: float = 0.001,
        momentum: float = 0,
        class_weights: Optional[torch.Tensor] = None,
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
        self.criterion = torch.nn.BCEWithLogitsLoss(pos_weight=class_weights)
        self.optimizer = torch.optim.SGD(self.linear.parameters(), lr=learning_rate, momentum=momentum)

        self.input_dim = input_dim
        self.output_dim = output_dim

    def forward(self, x: torch.Tensor):
        return self.linear(x)

    def update_weights(
            self,
            x: torch.Tensor,
            gradients: torch.Tensor,
            is_single: bool = False,
            collected_from_arbiter: bool = False
    ) -> None:
        if collected_from_arbiter:
            updated_weight = self.linear.weight.data.clone() - gradients.T
            self.linear.weight.data = updated_weight

        else:
            self.optimizer.zero_grad()
            logit = self.forward(x)
            if is_single:
                loss = self.criterion(torch.squeeze(logit), gradients.float())
                loss.backward()
            else:
                logit.backward(gradient=gradients)
            self.optimizer.step()

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)

    def get_weights(self) -> torch.Tensor:
        return self.linear.weight.clone()


    @property
    def init_params(self):
        return {
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
        }
