import logging
from collections import OrderedDict
from typing import List
from typing import Optional
from typing import Union

import torch
from torch import nn, Tensor
from torchsummary import summary

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
        **kwargs,
    ):
        super(ResNetTop, self).__init__()

        num_features = input_dim if num_init_features is None else num_init_features

        self.features = nn.Sequential(OrderedDict([]))
        if use_bn:
            self.features.add_module("norm", nn.BatchNorm1d(num_features))

        self.features.add_module("act", act_fun())
        self.fc = nn.Linear(num_features, output_dim)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                if init_weights:
                    nn.init.constant_(m.weight, init_weights)
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
            loss.backward()
            optimizer.step()
            return grads[0]
        else:
            model_output = self.forward(x)
            model_output.backward(gradient=gradients)
            # x.backward(gradient=gradients) #todo: rewise
            optimizer.step()

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)


if __name__ == "__main__":
    model = ResNetTop(input_dim=200, output_dim=10, use_bn=True, hid_factor=[0.1, 0.1])
    summary(model, input_size=(200,), device="cpu", batch_size=3)
