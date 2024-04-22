import logging
from collections import OrderedDict
from typing import List
from typing import Optional
from typing import Union

import torch
import torch.nn as nn

from stalactite.models.resnet import ResNetBlock
from stalactite.utils import init_linear_np

logger = logging.getLogger(__name__)


class ResNetBottom(nn.Module):
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
        hid_factor: List[float] = [2, 2],
        drop_rate: Union[float, List[float], List[List[float]]] = 0.1,
        noise_std: float = 0.05,
        act_fun: nn.Module = nn.ReLU,
        num_init_features: Optional[int] = None,
        use_bn: bool = True,
        use_noise: bool = False,
        device: torch.device = torch.device("cpu"),
        init_weights: float = None,
        seed: int = None,
        **kwargs,
    ):
        super(ResNetBottom, self).__init__()
        if isinstance(drop_rate, float):
            drop_rate = [[drop_rate, drop_rate]] * len(hid_factor)
        elif isinstance(drop_rate, list) and len(drop_rate) == 2:
            drop_rate = [drop_rate] * len(hid_factor)
        else:
            assert (
                len(drop_rate) == len(hid_factor) and len(drop_rate[0]) == 2
            ), "Wrong number hidden_sizes/drop_rates. Must be equal."
        self.seed = seed
        num_features = input_dim if num_init_features is None else num_init_features
        self.dense0 = nn.Linear(input_dim, num_features) if num_init_features is not None else nn.Identity()
        self.features1 = nn.Sequential(OrderedDict([]))

        for i, hd_factor in enumerate(hid_factor):
            block = ResNetBlock(
                n_in=num_features,
                hid_factor=hd_factor,
                n_out=num_features,
                drop_rate=drop_rate[i],
                noise_std=noise_std,
                act_fun=act_fun,
                use_bn=use_bn,
                use_noise=use_noise,
                device=device,
            )
            self.features1.add_module("resnetblock%d" % (i + 1), block)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                if init_weights:
                    nn.init.constant_(m.weight, init_weights)
                else:
                    init_linear_np(m, seed=self.seed)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward-pass."""
        x = self.dense0(x)
        identity = x
        for name, layer in self.features1.named_children():
            if name != "resnetblock1":
                x += identity
                identity = x
            x = layer(x)

        return x

    def update_weights(self, x: torch.Tensor, gradients: torch.Tensor, criterion=None, is_single: bool = True,
                       optimizer: torch.optim.Optimizer = None) -> None:
        optimizer.zero_grad()
        features = self.forward(x)
        features.backward(gradient=gradients)
        optimizer.step()

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)

