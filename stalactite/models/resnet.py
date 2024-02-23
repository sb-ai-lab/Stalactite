import logging
from collections import OrderedDict
from typing import List
from typing import Optional
from typing import Union

import torch
import torch.nn as nn
from torchsummary import summary

logger = logging.getLogger(__name__)

class GaussianNoise(nn.Module):
    """Adds gaussian noise.

    Args:
        stddev: Std of noise.
        device: Device to compute on.

    """

    def __init__(self, stddev: float, device: torch.device):
        super().__init__()
        self.stddev = stddev
        self.device = device

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward-pass."""
        if self.training:
            return x + torch.randn(x.size(), device=self.device) * self.stddev
        return x


class ResNetBlock(nn.Module):
    """Realisation of `'resnet'` model block.

    Args:
            n_in: Input dim.
            n_out: Output dim.
            hid_factor: Dim of intermediate fc is increased times this factor in ResnetModel layer.
            drop_rate: Dropout rates.
            noise_std: Std of noise.
            act_fun: Activation function.
            use_bn: Use BatchNorm.
            use_noise: Use noise.
            device: Device to compute on.

    """

    def __init__(
        self,
        n_in: int,
        hid_factor: float,
        n_out: int,
        drop_rate: List[float] = [0.1, 0.1],
        noise_std: float = 0.05,
        act_fun: nn.Module = nn.ReLU,
        use_bn: bool = True,
        use_noise: bool = False,
        device: torch.device = torch.device("cpu"),
        **kwargs,
    ):
        super(ResNetBlock, self).__init__()
        self.features = nn.Sequential(OrderedDict([]))

        if use_bn:
            self.features.add_module("norm", nn.BatchNorm1d(n_in))
        if use_noise:
            self.features.add_module("noise", GaussianNoise(noise_std, device))

        self.features.add_module("dense1", nn.Linear(n_in, int(hid_factor * n_in)))
        self.features.add_module("act1", act_fun())

        if drop_rate[0]:
            self.features.add_module("drop1", nn.Dropout(p=drop_rate[0]))

        self.features.add_module("dense2", nn.Linear(int(hid_factor * n_in), n_out))

        if drop_rate[1]:
            self.features.add_module("drop2", nn.Dropout(p=drop_rate[1]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward-pass."""
        x = self.features(x)
        return x


class ResNet(nn.Module):
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
        hid_factor: List[float] = [2, 2],
        drop_rate: Union[float, List[float], List[List[float]]] = 0.1,
        noise_std: float = 0.05,
        act_fun: nn.Module = nn.ReLU,
        num_init_features: Optional[int] = None,
        use_bn: bool = True,
        use_noise: bool = False,
        device: torch.device = torch.device("cpu"),
        init_weights: float = None,
        **kwargs,
    ):
        super(ResNet, self).__init__()
        if isinstance(drop_rate, float):
            drop_rate = [[drop_rate, drop_rate]] * len(hid_factor)
        elif isinstance(drop_rate, list) and len(drop_rate) == 2:
            drop_rate = [drop_rate] * len(hid_factor)
        else:
            assert (
                len(drop_rate) == len(hid_factor) and len(drop_rate[0]) == 2
            ), "Wrong number hidden_sizes/drop_rates. Must be equal."

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

        self.features2 = nn.Sequential(OrderedDict([]))
        if use_bn:
            self.features2.add_module("norm", nn.BatchNorm1d(num_features))

        self.features2.add_module("act", act_fun())
        self.fc = nn.Linear(num_features, output_dim)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                if init_weights:
                    nn.init.constant_(m.weight, init_weights)
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

        x = self.features2(x)
        x = self.fc(x)
        return x.view(x.shape[0], -1)

    def update_weights(self, x: torch.Tensor, y: torch.Tensor, criterion, is_single: bool,
                       optimizer: torch.optim.Optimizer = None) -> None:
        optimizer.zero_grad()
        logit = self.forward(x)
        loss = criterion(torch.squeeze(logit), y.type(torch.FloatTensor))
        logger.info(f"loss: {loss.item()}")
        loss.backward()
        optimizer.step()

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)


if __name__ == "__main__":
    model = ResNet(input_dim=200, output_dim=10, use_bn=True, hid_factor=[0.1, 0.1])
    summary(model, input_size=(200,), device="cpu", batch_size=3)
