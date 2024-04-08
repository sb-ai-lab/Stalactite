import logging
from typing import List

import mlflow
import torch
from torch import nn
from sklearn.metrics import roc_auc_score


from stalactite.models.split_learning import EfficientNetTop, MLPTop, ResNetTop
from stalactite.ml.honest.split_learning.base import HonestPartyMasterSplitNN
from stalactite.base import DataTensor, PartyDataTensor

logger = logging.getLogger(__name__)


class HonestPartyMasterEfficientNetSplitNN(HonestPartyMasterSplitNN):

    def initialize_model(self, do_load_model: bool = False) -> None:
        """ Initialize the model based on the specified model name. """
        self._model = EfficientNetTop(**self._model_params)
        class_weights = None if self.class_weights is None else self.class_weights.type(torch.FloatTensor)
        self._criterion = nn.CrossEntropyLoss(weight=class_weights)
        self._activation = nn.Softmax(dim=1)

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
        self.check_if_ready()

        for member_id, member_prediction in zip(participating_members, party_predictions):
            self.party_predictions[member_id] = member_prediction
        party_predictions = list(self.party_predictions.values())
        predictions = torch.mean(torch.stack(party_predictions, dim=1), dim=1)
        return predictions

    def report_metrics(self, y: DataTensor, predictions: DataTensor, name: str, step: int) -> None:
        postfix = "-infer" if step == -1 else ""
        step = step if step != -1 else None

        avg = "macro"
        roc_auc = roc_auc_score(y, predictions, average=avg, multi_class="ovr")
        logger.info(f'{name} ROC AUC {avg} on step {step}: {roc_auc}')
        if self.run_mlflow:
            mlflow.log_metric(f"{name.lower()}_roc_auc_{avg}{postfix}", roc_auc, step=step)