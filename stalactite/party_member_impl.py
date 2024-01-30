import logging
from typing import List, Optional

import scipy as sp
import torch
from datasets.dataset_dict import DatasetDict

from stalactite.base import Batcher, DataTensor, PartyMember, RecordsBatch
from stalactite.batching import ListBatcher
from stalactite.data_loader import AttrDict

logger = logging.getLogger(__name__)


class PartyMemberImpl(PartyMember):
    def __init__(
        self,
        uid: str,
        epochs: int,
        batch_size: int,
        model_update_dim_size: int,
        member_record_uids: List[str],
        model: torch.nn.Module,
        dataset: DatasetDict,
        data_params: AttrDict,
        report_train_metrics_iteration: int,
        report_test_metrics_iteration: int,
    ):
        self.id = uid
        self.epochs = epochs
        self._batch_size = batch_size
        self._uids = member_record_uids
        self._uids_to_use: Optional[List[str]] = None
        self.is_initialized = False
        self.is_finalized = False
        self._weights: Optional[DataTensor] = None
        self._weights_dim = model_update_dim_size
        self._data: Optional[DataTensor] = None
        self.iterations_counter = 0
        self._model = model
        self._dataset = dataset
        self._data_params = data_params
        self.report_train_metrics_iteration = report_train_metrics_iteration
        self.report_test_metrics_iteration = report_test_metrics_iteration

        self._batcher = None

    def _create_batcher(self, epochs: int, uids: List[str], batch_size: int) -> Batcher:
        logger.info("Member %s: making a batcher for uids" % (self.id))
        self._check_if_ready()
        self._batcher = ListBatcher(epochs=epochs, members=None, uids=uids, batch_size=batch_size)

    @property
    def batcher(self) -> Batcher:
        if self._batcher is None:
            if self._uids_to_use is None:
                raise RuntimeError("Cannot create batcher, you must `register_records_uids` first.")
            self._create_batcher(epochs=self.epochs, uids=self._uids_to_use, batch_size=self._batch_size)
        else:
            logger.info("Member %s: using created batcher" % (self.id))
        return self._batcher

    def records_uids(self) -> List[str]:
        logger.info("Member %s: reporting existing record uids" % self.id)
        return self._uids

    def register_records_uids(self, uids: List[str]):
        logger.info("Member %s: registering %s uids to be used." % (self.id, len(uids)))
        self._uids_to_use = uids

    def initialize(self):
        logger.info("Member %s: initializing" % self.id)
        self._weights = torch.rand(self._weights_dim)
        self._data = torch.rand(len(self._uids_to_use), self._weights_dim)
        self.is_initialized = True
        logger.info("Member %s: has been initialized" % self.id)

    def finalize(self):
        logger.info("Member %s: finalizing" % self.id)
        self._check_if_ready()
        self._weights = None
        self.is_finalized = True
        logger.info("Member %s: has been finalized" % self.id)

    def _prepare_data(self, uids: RecordsBatch):
        X_train = self._dataset[self._data_params.train_split][self._data_params.features_key][[int(x) for x in uids]]
        U, S, Vh = sp.linalg.svd(X_train.numpy(), full_matrices=False, overwrite_a=False, check_finite=False)
        return U, S, Vh

    def update_weights(self, uids: RecordsBatch, upd: DataTensor):
        logger.info("Member %s: updating weights. Incoming tensor: %s" % (self.id, tuple(upd.size())))
        self._check_if_ready()
        X_train = self._dataset[self._data_params.train_split][self._data_params.features_key][[int(x) for x in uids]]
        self._model.update_weights(X_train, upd)
        logger.info("Member %s: successfully updated weights" % self.id)

    def predict(self, uids: RecordsBatch, use_test: bool = False) -> DataTensor:
        logger.info("Member %s: predicting. Batch size: %s" % (self.id, len(uids)))
        self._check_if_ready()
        if use_test:
            logger.info("Member %s: using test data" % self.id)
            X = self._dataset[self._data_params.test_split][self._data_params.features_key]
        else:
            X = self._dataset[self._data_params.train_split][self._data_params.features_key][[int(x) for x in uids]]
        predictions = self._model.predict(X)
        logger.info("Member %s: made predictions." % self.id)
        return predictions

    def update_predict(self, upd: DataTensor, previous_batch: RecordsBatch, batch: RecordsBatch) -> DataTensor:

        logger.info("Member %s: updating and predicting." % self.id)
        self._check_if_ready()
        uids = previous_batch if previous_batch is not None else batch
        self.update_weights(uids=uids, upd=upd)
        predictions = self.predict(batch)
        self.iterations_counter += 1
        logger.info("Member %s: updated and predicted." % self.id)
        return predictions

    def _check_if_ready(self):
        if not self.is_initialized and not self.is_finalized:
            raise RuntimeError("The member has not been initialized")
