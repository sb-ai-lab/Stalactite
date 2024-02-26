from typing import Any

from stalactite.ml.honest.linear_regression.party_member import HonestPartyMemberLinReg
from stalactite.models import LogisticRegressionBatch


class HonestPartyMemberLogReg(HonestPartyMemberLinReg):
    def initialize_model_from_params(self, **model_params) -> Any:
        return LogisticRegressionBatch(**model_params)

    def initialize_model(self, do_load_model: bool = False) -> None:
        """ Initialize the model based on the specified model name. """
        if do_load_model:
            self._model = self.load_model()
        else:
            self._model = LogisticRegressionBatch(
                input_dim=self._dataset[self._data_params.train_split][self._data_params.features_key].shape[1],
                output_dim=self._dataset[self._data_params.train_split][self._data_params.label_key].shape[1],
                learning_rate=self._common_params.learning_rate,
                class_weights=None,
                init_weights=0.005
            )
