from typing import Any

from torch.optim import SGD

from stalactite.ml.honest.linear_regression.party_member import HonestPartyMemberLinReg
from stalactite.models.split_learning import ResNetBottom


class HonestPartyMemberResNet(HonestPartyMemberLinReg):

    def initialize_model(self, do_load_model: bool = False) -> None:
        """ Initialize the model based on the specified model name. """
        input_dim = self._dataset[self._data_params.train_split][self._data_params.features_key].shape[1]
        if do_load_model:
            self._model = self.load_model()
        else:
            self._model = ResNetBottom(input_dim=input_dim, **self._model_params)

    def initialize_optimizer(self) -> None:
        self._optimizer = SGD([
            {"params": self._model.parameters()},
        ],
            lr=self._common_params.learning_rate,
            momentum=self._common_params.momentum
        )
