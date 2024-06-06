import logging
from typing import List, Any

import mlflow
import torch
from torch import nn
from sklearn.metrics import roc_auc_score

from stalactite.models.split_learning import EfficientNetTop
from stalactite.ml.honest.split_learning.base import HonestPartyMasterSplitNN
from stalactite.base import DataTensor, PartyDataTensor

logger = logging.getLogger(__name__)


class HonestPartyMasterEfficientNetSplitNN(HonestPartyMasterSplitNN):

    def initialize_model_from_params(self, **model_params) -> Any:
        return EfficientNetTop(**model_params).to(self.device)

    def initialize_model(self, do_load_model: bool = False) -> None:
        """ Initialize the model based on the specified model name. """
        logger.info(f"Master {self.id} initializes model on device: {self.device}")
        logger.debug(f"Model is loaded from path: {do_load_model}")
        if do_load_model:
            self._model = self.load_model().to(self.device)
        else:
            self._model = EfficientNetTop(**self._model_params, seed=self.seed).to(self.device)
            class_weights = None if self.class_weights is None else self.class_weights.type(torch.FloatTensor) \
                .to(self.device)
            self._criterion = nn.CrossEntropyLoss(weight=class_weights)
        self._activation = nn.Softmax(dim=1)

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
        predictions = torch.mean(torch.stack(party_predictions, dim=1), dim=1)
        return predictions

    def report_metrics(self, y: DataTensor, predictions: DataTensor, name: str, step: int) -> None:
        logger.info(f"Master {self.id} reporting metrics")
        logger.debug(f"Predictions size: {predictions.size()}, Target size: {y.size()}")
        y = y.cpu().numpy()
        predictions = predictions.cpu().detach().numpy()

        postfix = "-infer" if step == -1 else ""
        step = step if step != -1 else None

        avg = "macro"
        roc_auc = roc_auc_score(y, predictions, average=avg, multi_class="ovr")
        logger.info(f'{name} ROC AUC {avg} on step {step}: {roc_auc}')
        if self.run_mlflow:
            mlflow.log_metric(f"{name.lower()}_roc_auc_{avg}{postfix}", roc_auc, step=step)
