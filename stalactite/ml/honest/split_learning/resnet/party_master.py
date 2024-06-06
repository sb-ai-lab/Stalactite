import logging
from typing import List, Any

import torch
from stalactite.models.split_learning import ResNetTop
from stalactite.ml.honest.split_learning.base import HonestPartyMasterSplitNN
from stalactite.base import DataTensor, PartyDataTensor

logger = logging.getLogger(__name__)


class HonestPartyMasterResNetSplitNN(HonestPartyMasterSplitNN):

    def initialize_model_from_params(self, **model_params) -> Any:
        return ResNetTop(**model_params).to(self.device)

    def initialize_model(self, do_load_model: bool = False) -> None:
        """ Initialize the model based on the specified model name. """
        logger.info(f"Master {self.id} initializes model on device: {self.device}")
        logger.debug(f"Model is loaded from path: {do_load_model}")
        if do_load_model:
            self._model = self.load_model().to(self.device)
        else:
            self._model = ResNetTop(**self._model_params, seed=self.seed).to(self.device)
            class_weights = None if self.class_weights is None else self.class_weights.type(torch.FloatTensor) \
                .to(self.device)
            self._criterion = torch.nn.BCEWithLogitsLoss(
                pos_weight=class_weights) if self.binary else torch.nn.CrossEntropyLoss(weight=class_weights)

    def initialize_optimizer(self) -> None:
        self._optimizer = torch.optim.SGD([
            {"params": self._model.parameters()},
        ],
            lr=self._common_params.learning_rate,
            momentum=self._common_params.momentum,
            weight_decay=self._common_params.weight_decay,

        )

    def aggregate(
            self, participating_members: List[str], party_predictions: PartyDataTensor, is_infer: bool = False
    ) -> DataTensor:
        logger.info(f"Master {self.id}: aggregates party predictions (number of predictions {len(party_predictions)})")
        self.check_if_ready()
        for member_id, member_prediction in zip(participating_members, party_predictions):
            self.party_predictions[member_id] = member_prediction.to(self.device)
        party_predictions = list(self.party_predictions.values())
        predictions = torch.cat(party_predictions, dim=1)
        return predictions
