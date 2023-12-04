import logging
from typing import List, Optional

import torch
from sklearn import metrics

from stalactite.base import PartyMaster, DataTensor, Batcher, PartyDataTensor, PartyMember
from stalactite.batching import ListBatcher

logger = logging.getLogger(__name__)


class MockPartyMasterImpl(PartyMaster):
    def __init__(self,
                 uid: str,
                 epochs: int,
                 report_train_metrics_iteration: int,
                 report_test_metrics_iteration: int,
                 target: DataTensor):
        self.id = uid
        self.epochs = epochs
        self.report_train_metrics_iteration = report_train_metrics_iteration
        self.report_test_metrics_iteration = report_test_metrics_iteration
        self.target = target
        self._is_initialized = False
        self._is_finalized = False
        self._batch_size = 10
        self._weights_dim = 100

    def master_initialize(self):
        logger.info("Master %s: initializing" % self.id)
        self._is_initialized = True

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
        return torch.mean(torch.stack(party_predictions))

    def compute_updates(self, predictions: DataTensor, party_predictions: PartyDataTensor, world_size: int) \
            -> List[DataTensor]:
        logger.info("Master %s: computing updates (world size %s)" % (self.id, world_size))
        self._check_if_ready()
        return [predictions + torch.rand(self._weights_dim) for _ in range(world_size)]

    def master_finalize(self):
        logger.info("Master %s: finalizing" % self.id)
        self._check_if_ready()
        self._is_finalized = True

    def _check_if_ready(self):
        if not self._is_initialized and not self._is_finalized:
            raise RuntimeError("The member has not been initialized")


class MockPartyMemberImpl(PartyMember):
    def __init__(self, uid: str):
        self.id = uid
        self._uids = [str(i) for i in range(100)]
        self._uids_to_use: Optional[List[str]] = None
        self._is_initialized = False
        self._is_finalized = False
        self._weights: Optional[DataTensor] = None
        self._weights_dim = 100
        self._data: Optional[DataTensor] = None

    def records_uids(self) -> List[str]:
        logger.info("Member %s: reporting existing record uids" % self.id)
        return self._uids

    def register_records_uids(self, uids: List[str]):
        logger.info("Member %s: registering uids to be used: %s" % (self.id, uids))
        self._uids_to_use = uids

    def initialize(self):
        logger.info("Member %s: initializing" % self.id)
        self._weights = torch.rand(self._weights_dim)
        self._data = torch.rand(len(self._uids_to_use), self._weights_dim)
        self._is_initialized = True
        logger.info("Member %s: has been initialized" % self.id)

    def finalize(self):
        logger.info("Member %s: finalizing" % self.id)
        self._check_if_ready()
        self._weights = None
        self._is_finalized = True
        logger.info("Member %s: has been finalized" % self.id)

    def update_weights(self, upd: DataTensor):
        logger.info("Member %s: updating weights. Incoming tensor: %s" % (self.id, tuple(upd.size())))
        self._check_if_ready()
        if upd.size() != self._weights.size():
            raise ValueError(f"Incorrect size of update. "
                             f"Expected: {tuple(upd.size())}. Actual: {tuple(self._weights.size())}")

        self._weights += upd
        logger.info("Member %s: successfully updated weights" % self.id)

    def predict(self, batch: List[str]) -> DataTensor:
        logger.info("Member %s: predicting. Batch: %s" % (self.id, batch))
        self._check_if_ready()
        batch = set(batch)
        idx = [i for i, uid in enumerate(self._uids_to_use) if uid in batch]
        predictions = self._data[idx, :] * self._weights
        logger.info("Member %s: made predictions." % self.id)
        return predictions

    def update_predict(self, batch: List[str], upd: DataTensor) -> DataTensor:
        logger.info("Member %s: updating and predicting." % self.id)
        self._check_if_ready()
        self.update_weights(upd)
        predictions = self.predict(batch)
        logger.info("Member %s: updated and predicted." % self.id)
        return predictions

    def _check_if_ready(self):
        if not self._is_initialized and not self._is_finalized:
            raise RuntimeError("The member has not been initialized")
