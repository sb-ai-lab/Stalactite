import logging
from collections import OrderedDict
from typing import Optional

import torch
from torch import nn, Tensor

from stalactite.utils import init_linear_np

logger = logging.getLogger(__name__)


class ResNetTop(nn.Module):
    """The ResNet model from https://github.com/Yura52/rtdl.

    Args:
            n_in: Input dim.
            n_out: Output dim.
            hid_factor: Dim of intermediate fc is increased times this factor in ResnetModel layer.
            drop_rate: Dropout rate for each layer separately or altogether.
            noise_std: Std of noise.
            act_fun: Activation function.
            num_init_features: If not none add fc layer before model with certain dim.
            use_bn: Use BatchNorm.
            use_noise: Use noise.
            device: Device to compute on.

    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 1,
        act_fun: nn.Module = nn.ReLU,
        num_init_features: Optional[int] = None,
        use_bn: bool = True,
        init_weights: float = None,
        seed: int = None,
        **kwargs,
    ):
        super(ResNetTop, self).__init__()

        num_features = input_dim if num_init_features is None else num_init_features

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_init_features = num_init_features
        self.use_bn = use_bn
        self.init_weights = init_weights
        self.seed = seed

        self.features = nn.Sequential(OrderedDict([]))
        if use_bn:
            self.features.add_module("norm", nn.BatchNorm1d(num_features))

        self.features.add_module("act", act_fun())
        self.fc = nn.Linear(num_features, output_dim)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                if init_weights:
                    nn.init.constant_(m.weight, init_weights)
                else:
                    init_linear_np(m, seed=self.seed)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward-pass."""
        x = self.features(x)
        x = self.fc(x)
        return x.view(x.shape[0], -1)

    def update_weights(self, x: torch.Tensor, gradients: torch.Tensor, criterion=None, is_single: bool = False,
                       optimizer: torch.optim.Optimizer = None) -> Optional[Tensor]:
        optimizer.zero_grad()

        if is_single:
            logit = self.forward(x)
            loss = criterion(torch.squeeze(logit), gradients.type(torch.FloatTensor))
            grads = torch.autograd.grad(outputs=loss, inputs=x, retain_graph=True)
            logger.info(f"Loss: {loss.item()}")
            loss.backward()
            optimizer.step()
            return grads[0]
        else:
            model_output = self.forward(x)
            model_output.backward(gradient=gradients)
            optimizer.step()

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)

    @property
    def init_params(self):
        return {
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'num_init_features': self.num_init_features,
            'use_bn': self.use_bn,
            'seed': self.seed,
            'init_weights': self.init_weights,
        }
