import logging
from typing import List

import torch
from torch import nn
from stalactite.models.split_learning import ResNetTop
from stalactite.ml.honest.split_learning.base import HonestPartyMasterSplitNN
from stalactite.base import DataTensor, PartyDataTensor

logger = logging.getLogger(__name__)


class HonestPartyMasterResNetSplitNN(HonestPartyMasterSplitNN):

    def initialize_model(self, do_load_model: bool = False) -> None:
        """ Initialize the model based on the specified model name. """
        self._model = ResNetTop(**self._model_params)
        self._criterion = nn.BCEWithLogitsLoss(pos_weight=self.class_weights)
        self._activation = nn.Sigmoid()

    def initialize_optimizer(self) -> None:
        self._optimizer = torch.optim.SGD([
            {"params": self._model.parameters()},
        ],
            lr=self._common_params.learning_rate,
            momentum=self._common_params.momentum
        )

    def aggregate(
            self, participating_members: List[str], party_predictions: PartyDataTensor, is_infer: bool = False
    ) -> DataTensor:
        logger.info("Master %s: aggregating party predictions (num predictions %s)" % (self.id, len(party_predictions)))
        self._check_if_ready()

        for member_id, member_prediction in zip(participating_members, party_predictions):
            self.party_predictions[member_id] = member_prediction
        party_predictions = list(self.party_predictions.values())
        predictions = torch.cat(party_predictions, dim=1)
        return predictions
