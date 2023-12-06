import logging
from typing import List, Optional

import torch
from sklearn import metrics

from stalactite.base import PartyMaster, DataTensor, Batcher, PartyDataTensor, PartyMember, Party
from stalactite.batching import ListBatcher

logger = logging.getLogger(__name__)

class PartyMasterImpl(PartyMaster):
    def __init__(self,
                 uid: str,
                 epochs: int,
                 report_train_metrics_iteration: int,
                 report_test_metrics_iteration: int,
                 target: DataTensor,
                 target_uids: List[str],
                 batch_size: int,
                 model_update_dim_size: int):
        self.id = uid
        self.epochs = epochs
        self.report_train_metrics_iteration = report_train_metrics_iteration
        self.report_test_metrics_iteration = report_test_metrics_iteration
        self.target = target
        self.target_uids = target_uids
        self.is_initialized = False
        self.is_finalized = False
        self._batch_size = batch_size
        self._weights_dim = model_update_dim_size
        self.iteration_counter = 0

    def master_initialize(self, party: Party):
        logger.info("Master %s: initializing" % self.id)
        party.initialize()
        self.is_initialized = True

    def make_batcher(self, uids: List[str]) -> Batcher:
        logger.info("Master %s: making a batcher for uids %s" % (self.id, uids))
        self._check_if_ready()
        return ListBatcher(uids=uids, batch_size=self._batch_size)

    def make_init_updates(self, world_size: int) -> PartyDataTensor:
        logger.info("Master %s: making init updates for %s members" % (self.id, world_size))
        self._check_if_ready()
        return [torch.rand(self._weights_dim) for _ in range(world_size)]

    def report_metrics(self, y: DataTensor, predictions: DataTensor, name: str):
        logger.info(f"Master %s: reporting metrics. Y dim: {y.size()}. "
                    f"Predictions size: {predictions.size()}" % self.id)
        error = metrics.mean_absolute_error(y, predictions)
        logger.info(f"Master %s: mock metrics (MAE): {error}" % error)

    def aggregate(self, party_predictions: PartyDataTensor) -> DataTensor:
        logger.info("Master %s: aggregating party predictions (num predictions %s)" % (self.id, len(party_predictions)))
        self._check_if_ready()
        # assert all(prediction.size() == (self._batch_size for prediction in party_predictions)
        return torch.mean(torch.stack(party_predictions, dim=1), dim=1)

    def compute_updates(self, predictions: DataTensor, party_predictions: PartyDataTensor, world_size: int) \
            -> List[DataTensor]:
        logger.info("Master %s: computing updates (world size %s)" % (self.id, world_size))
        self._check_if_ready()
        self.iteration_counter += 1
        return [torch.rand(self._weights_dim) for _ in range(world_size)]

    def master_finalize(self, party: Party):
        logger.info("Master %s: finalizing" % self.id)
        self._check_if_ready()
        party.finalize()
        self.is_finalized = True

    def _check_if_ready(self):
        if not self.is_initialized and not self.is_finalized:
            raise RuntimeError("The member has not been initialized")