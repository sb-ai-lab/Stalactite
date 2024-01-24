import logging
from typing import List

import mlflow
import torch
from sklearn import metrics

from stalactite.base import Batcher, DataTensor, Party, PartyDataTensor, PartyMaster
from stalactite.batching import ConsecutiveListBatcher, ListBatcher
from stalactite.metrics import ComputeAccuracy

logger = logging.getLogger(__name__)


class PartyMasterImpl(PartyMaster):
    def __init__(
        self,
        uid: str,
        epochs: int,
        report_train_metrics_iteration: int,
        report_test_metrics_iteration: int,
        target: DataTensor,
        test_target: DataTensor,
        target_uids: List[str],
        batch_size: int,
        model_update_dim_size: int,
        run_mlflow: bool = False,
    ):
        self.id = uid
        self.epochs = epochs
        self.report_train_metrics_iteration = report_train_metrics_iteration
        self.report_test_metrics_iteration = report_test_metrics_iteration
        self.target = target
        self.test_target = test_target
        self.target_uids = target_uids
        self.is_initialized = False
        self.is_finalized = False
        self._batch_size = batch_size
        self._weights_dim = model_update_dim_size
        self.iteration_counter = 0
        self.run_mlflow = run_mlflow
        self.party_predictions = dict()
        self.updates = dict()

    def master_initialize(self, party: Party):
        logger.info("Master %s: initializing" % self.id)
        party.initialize()
        self.is_initialized = True

    def make_batcher(self, uids: List[str], party: Party) -> Batcher:
        logger.info("Master %s: making a batcher for uids %s" % (self.id, uids))
        self._check_if_ready()
        return ListBatcher(epochs=self.epochs, members=party.members, uids=uids, batch_size=self._batch_size)

    def make_init_updates(self, world_size: int) -> PartyDataTensor:
        logger.info("Master %s: making init updates for %s members" % (self.id, world_size))
        self._check_if_ready()
        return [torch.rand(self._batch_size) for _ in range(world_size)]

    def report_metrics(self, y: DataTensor, predictions: DataTensor, name: str):
        logger.info(
            f"Master %s: reporting metrics. Y dim: {y.size()}. " f"Predictions size: {predictions.size()}" % self.id
        )
        mae = metrics.mean_absolute_error(y, predictions)
        acc = ComputeAccuracy().compute(y, predictions)
        logger.info(f"Master %s: %s metrics (MAE): {mae}" % (self.id, name))
        logger.info(f"Master %s: %s metrics (Accuracy): {acc}" % (self.id, name))

        if self.run_mlflow:
            step = self.iteration_counter
            mlflow.log_metric(f"{name.lower()}_mae", mae, step=step)
            mlflow.log_metric(f"{name.lower()}_acc", acc, step=step)

    def aggregate(
        self, participating_members: List[str], party_predictions: PartyDataTensor, infer=False
    ) -> DataTensor:
        logger.info("Master %s: aggregating party predictions (num predictions %s)" % (self.id, len(party_predictions)))
        self._check_if_ready()
        if not infer:
            for member_id, member_prediction in zip(participating_members, party_predictions):
                self.party_predictions[member_id] = member_prediction
            party_predictions = list(self.party_predictions.values())

        return torch.sum(torch.stack(party_predictions, dim=1), dim=1)

    def compute_updates(
        self,
        participating_members: List[str],
        predictions: DataTensor,
        party_predictions: PartyDataTensor,
        world_size: int,
        subiter_seq_num: int,
    ) -> List[DataTensor]:

        logger.info("Master %s: computing updates (world size %s)" % (self.id, world_size))
        self._check_if_ready()
        self.iteration_counter += 1
        y = self.target[self._batch_size * subiter_seq_num : self._batch_size * (subiter_seq_num + 1)]

        for member_id in participating_members:
            party_predictions_for_upd = [v for k, v in self.party_predictions.items() if k != member_id]
            if len(party_predictions_for_upd) == 0:
                party_predictions_for_upd = [torch.rand(predictions.size())]
            pred_for_member_upd = torch.mean(torch.stack(party_predictions_for_upd), dim=0)
            member_update = y - torch.reshape(pred_for_member_upd, (-1,))
            self.updates[member_id] = member_update

        return [self.updates[member_id] for member_id in participating_members]

    def master_finalize(self, party: Party):
        logger.info("Master %s: finalizing" % self.id)
        self._check_if_ready()
        party.finalize()
        self.is_finalized = True

    def _check_if_ready(self):
        if not self.is_initialized and not self.is_finalized:
            raise RuntimeError("The member has not been initialized")


class PartyMasterImplConsequently(PartyMasterImpl):
    def make_batcher(self, uids: List[str], party: Party) -> Batcher:
        logger.info("Master %s: making a batcher for uids %s" % (self.id, uids))
        self._check_if_ready()
        return ConsecutiveListBatcher(epochs=self.epochs, members=party.members, uids=uids, batch_size=self._batch_size)
