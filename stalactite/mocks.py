import logging
from typing import List, Optional

import torch
from sklearn import metrics

from stalactite.base import (
    Batcher,
    DataTensor,
    PartyDataTensor,
    PartyMaster,
    PartyMember,
    RecordsBatch,
)
from stalactite.batching import ListBatcher

logger = logging.getLogger(__name__)


class MockPartyMasterImpl(PartyMaster):  # TODO
    def __init__(
            self,
            uid: str,
            epochs: int,
            report_train_metrics_iteration: int,
            report_test_metrics_iteration: int,
            target: DataTensor,
            target_uids: List[str],
            batch_size: int,
            model_update_dim_size: int,
    ):
        self.id = uid
        self.epochs = epochs
        self.report_train_metrics_iteration = report_train_metrics_iteration
        self.report_test_metrics_iteration = report_test_metrics_iteration
        self.target = target
        self.target_uids = target_uids
        self.test_target = target
        self.is_initialized = False
        self.is_finalized = False
        self._batch_size = batch_size
        self._weights_dim = model_update_dim_size
        self.iteration_counter = 0

    def initialize(self):
        logger.info("Master %s: initializing" % self.id)
        self.is_initialized = True

    def make_batcher(self, uids: List[str], party_members: List[str]) -> Batcher:
        logger.info("Master %s: making a make_batcher for uids %s" % (self.id, uids))
        self._check_if_ready()
        return ListBatcher(epochs=self.epochs, members=party_members, uids=uids, batch_size=self._batch_size)

    def make_init_updates(self, world_size: int) -> PartyDataTensor:
        logger.info("Master %s: making init updates for %s members" % (self.id, world_size))
        self._check_if_ready()
        return [torch.rand(self._weights_dim) for _ in range(world_size)]

    def report_metrics(self, y: DataTensor, predictions: DataTensor, name: str):
        logger.info(
            f"Master %s: reporting metrics. Y dim: {y.size()}. " f"Predictions size: {predictions.size()}" % self.id
        )
        error = metrics.mean_absolute_error(y, predictions)
        logger.info(f"Master %s: mock metrics (MAE): {error}" % error)

    def aggregate(
            self, participating_members: List[str], party_predictions: PartyDataTensor, infer: bool = False
    ) -> DataTensor:
        logger.info("Master %s: aggregating party predictions (num predictions %s)" % (self.id, len(party_predictions)))
        self._check_if_ready()
        return torch.mean(torch.stack(party_predictions, dim=1), dim=1)

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
        return [torch.rand(self._weights_dim) for _ in range(world_size)]

    def finalize(self):
        logger.info("Master %s: finalizing" % self.id)
        self._check_if_ready()
        self.is_finalized = True

    def _check_if_ready(self):
        if not self.is_initialized and not self.is_finalized:
            raise RuntimeError("The member has not been initialized")


class MockPartyMemberImpl(PartyMember):
    def __init__(
            self,
            uid: str,
            model_update_dim_size: int,
            member_record_uids: List[str],
            epochs: int,
            batch_size: int,
            report_train_metrics_iteration: int,
            report_test_metrics_iteration: int,
    ):
        self.id = uid
        self._uids = member_record_uids
        self._uids_to_use: Optional[List[str]] = None
        self.is_initialized = False
        self.is_finalized = False
        self._weights: Optional[DataTensor] = None
        self._weights_dim = model_update_dim_size
        self._data: Optional[DataTensor] = None
        self.iterations_counter = 0
        self.epochs = epochs
        self._batch_size = batch_size
        self._batcher: Optional[Batcher] = None
        self.report_train_metrics_iteration = report_train_metrics_iteration
        self.report_test_metrics_iteration = report_test_metrics_iteration

    def records_uids(self) -> List[str]:
        logger.info("Member %s: reporting existing record uids" % self.id)
        return self._uids

    def register_records_uids(self, uids: List[str]):
        logger.info("Member %s: registering uids to be used: %s" % (self.id, uids))
        self._uids_to_use = uids
        self._data = torch.rand(len(self._uids_to_use), self._weights_dim)

    def initialize(self):
        logger.info("Member %s: initializing" % self.id)
        self._weights = torch.rand(self._weights_dim)
        self.is_initialized = True
        logger.info("Member %s: has been initialized" % self.id)

    def finalize(self):
        logger.info("Member %s: finalizing" % self.id)
        self._check_if_ready()
        self._weights = None
        self.is_finalized = True
        logger.info("Member %s: has been finalized" % self.id)

    def update_weights(self, uids: RecordsBatch, upd: DataTensor):
        logger.info("Member %s: updating weights. Incoming tensor: %s" % (self.id, tuple(upd.size())))
        self._check_if_ready()
        if upd.size() != self._weights.size():
            raise ValueError(
                f"Incorrect size of update. " f"Expected: {tuple(self._weights.size())}. Actual: {tuple(upd.size())}"
            )

        self._weights += upd
        logger.info("Member %s: successfully updated weights" % self.id)

    def predict(self, uids: List[str], use_test: bool = False) -> DataTensor:
        logger.info("Member %s: predicting. Batch: %s" % (self.id, uids))
        self._check_if_ready()
        uids = set(uids)
        idx = [i for i, uid in enumerate(self._uids_to_use) if uid in uids]
        predictions = torch.sum(self._data[idx, :] * self._weights, dim=1)
        logger.info("Member %s: made predictions." % self.id)
        return predictions

    def update_predict(self, upd: DataTensor, batch: RecordsBatch, previous_batch: RecordsBatch) -> DataTensor:
        logger.info("Member %s: updating and predicting." % self.id)
        self._check_if_ready()
        self.update_weights(previous_batch, upd)
        predictions = self.predict(batch)
        self.iterations_counter += 1
        logger.info("Member %s: updated and predicted." % self.id)
        return predictions

    def _check_if_ready(self):
        if not self.is_initialized and not self.is_finalized:
            raise RuntimeError("The member has not been initialized")

    def _create_batcher(self, epochs: int, uids: List[str], batch_size: int) -> None:
        logger.info("Member %s: making a make_batcher for uids" % (self.id))
        self._check_if_ready()
        self._batcher = ListBatcher(epochs=epochs, members=None, uids=uids, batch_size=batch_size)

    @property
    def make_batcher(self) -> Batcher:
        if self._batcher is None:
            if self._uids_to_use is None:
                raise RuntimeError("Cannot create make_batcher, you must `register_records_uids` first.")
            self._create_batcher(epochs=self.epochs, uids=self._uids_to_use, batch_size=self._batch_size)
        else:
            logger.info("Member %s: using created make_batcher" % (self.id))
        return self._batcher
