import logging
from typing import Any

from torch.optim import SGD

from stalactite.ml.honest.linear_regression.party_member import HonestPartyMemberLinReg
from stalactite.models.split_learning import MLPBottom

logger = logging.getLogger(__name__)


class HonestPartyMemberMLP(HonestPartyMemberLinReg):

    def initialize_model_from_params(self, **model_params) -> Any:
        return MLPBottom(**model_params).to(self.device)

    def initialize_model(self, do_load_model: bool = False) -> None:
        """ Initialize the model based on the specified model name. """
        logger.info(f"Member {self.id} initializes model on device: {self.device}")
        logger.debug(f"Model is loaded from path: {do_load_model}")
        input_dim = self._dataset[self._data_params.train_split][self._data_params.features_key].shape[1]
        if do_load_model:
            self._model = self.load_model().to(self.device)
        else:
            self._model = MLPBottom(input_dim=input_dim, **self._model_params, seed=self.seed).to(self.device)

    def initialize_optimizer(self) -> None:
        self._optimizer = SGD([
            {"params": self._model.parameters()},
        ],
            lr=self._common_params.learning_rate,
            momentum=self._common_params.momentum,
            weight_decay=self._common_params.weight_decay,

        )

    def move_model_to_device(self):
        # As the class is inherited from the Linear regression model, we need to skip this step with returning the model
        # to device after weights updates
        pass
