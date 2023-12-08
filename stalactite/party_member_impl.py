import logging
from typing import List, Optional

import torch
import scipy as sp
from datasets.dataset_dict import DatasetDict

from stalactite.base import DataTensor, PartyMember
from stalactite.data_loader import AttrDict

logger = logging.getLogger(__name__)


class PartyMemberImpl(PartyMember):
    def __init__(self, uid: str, model_update_dim_size: int, member_record_uids: List[str], model: torch.nn.Module,
                 dataset: DatasetDict, data_params: AttrDict):
        self.id = uid
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
        self.is_initialized = True
        logger.info("Member %s: has been initialized" % self.id)

    def finalize(self):
        logger.info("Member %s: finalizing" % self.id)
        self._check_if_ready()
        self._weights = None
        self.is_finalized = True
        logger.info("Member %s: has been finalized" % self.id)

    def _prepare_data(self, uids: List[str]):

        X_train = self._dataset[self._data_params.train_split][self._data_params.features_key][[int(x) for x in uids]]
        U, S, Vh = sp.linalg.svd(X_train.numpy(), full_matrices=False, overwrite_a=False, check_finite=False)
        return U, S, Vh

    def update_weights(self, uids: List[str], upd: DataTensor):
        logger.info("Member %s: updating weights. Incoming tensor: %s" % (self.id, tuple(upd.size())))
        self._check_if_ready()
        U, S, Vh = self._prepare_data(uids)
        self._model.update_weights(data_U=U, data_S=S, data_Vh=Vh, rhs=upd)
        logger.info("Member %s: successfully updated weights" % self.id)

    def predict(self, uids: List[str], use_test: bool = False) -> DataTensor:
        logger.info("Member %s: predicting. Batch: %s" % (self.id, uids))
        self._check_if_ready()
        tensor_idx = [int(x) for x in uids]  # todo: do it somewhere else
        if use_test:
            logger.info("Member %s: using test data" % self.id)
            X = self._dataset[self._data_params.test_split][self._data_params.features_key]
        else:
            X = self._dataset[self._data_params.train_split][self._data_params.features_key][tensor_idx]
        predictions = self._model.predict(X)
        logger.info("Member %s: made predictions." % self.id)
        return predictions

    def update_predict(self, upd: DataTensor, batch: List[str]) -> DataTensor:
        logger.info("Member %s: updating and predicting." % self.id)
        self._check_if_ready()
        self.update_weights(uids=batch, upd=upd)
        predictions = self.predict(batch)
        self.iterations_counter += 1
        logger.info("Member %s: updated and predicted." % self.id)
        return predictions

    def _check_if_ready(self):
        if not self.is_initialized and not self.is_finalized:
            raise RuntimeError("The member has not been initialized")
