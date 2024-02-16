from typing import List, Optional
import logging

import torch

from stalactite.base import RecordsBatch, DataTensor, Batcher
from stalactite.batching import ListBatcher
from stalactite.ml.arbitered.base import ArbiteredPartyMember, SecurityProtocol
from stalactite.models import LogisticRegressionBatch

logger = logging.getLogger(__name__)

logger.setLevel(logging.DEBUG)
sh = logging.StreamHandler()
sh.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
logger.addHandler(sh)


class ArbiteredPartyMemberLogReg(ArbiteredPartyMember):
    def __init__(
            self,
            uid: str,
            epochs: int,
            batch_size: int,
            member_record_uids: List[str],
            security_protocol: SecurityProtocol,
            processor=None,
    ) -> None:
        self.id = uid
        self.epochs = epochs
        self.batch_size = batch_size
        self._uids = member_record_uids
        self.processor = processor
        self.security_protocol = security_protocol
        self._batcher = None

    _model: Optional[LogisticRegressionBatch] = None

    def predict_partial(self, uids: RecordsBatch) -> DataTensor:
        # X = self._dataset[self._data_params.train_split][self._data_params.features_key][[int(x) for x in uids]]
        predictions = self.predict(uids, is_test=False)
        Xw = 0.25 * predictions
        return Xw

    def predict(self, uids: Optional[List[str]], is_test: bool = False) -> DataTensor:
        if not is_test:
            if uids is None:
                uids = self._uids_to_use
            X = self._dataset[self._data_params.train_split][self._data_params.features_key][[int(x) for x in uids]]
        else:
            X = self._dataset[self._data_params.test_split][self._data_params.features_key]
        return self._model.predict(X)

    def compute_gradient(self, aggregated_predictions_diff: DataTensor, uids: List[str]) -> DataTensor:
        X = self._dataset[self._data_params.train_split][self._data_params.features_key][[int(x) for x in uids]]
        g = torch.matmul(X.T, aggregated_predictions_diff) / X.shape[0]
        return g

    @property
    def batcher(self) -> Batcher:
        if self._batcher is None:
            if self._uids_to_use is None:
                raise RuntimeError("Cannot create batcher, you must `register_records_uids` first.")
            self._batcher = ListBatcher(
                epochs=self.epochs,
                members=None,
                uids=self._uids_to_use,
                batch_size=self.batch_size
            )
        else:
            logger.info("Member %s: using created batcher" % (self.id))
        return self._batcher

    def records_uids(self) -> List[str]:
        logger.info("Member %s: reporting existing record uids" % self.id)
        return self._uids

    def register_records_uids(self, uids: List[str]):
        logger.info("Member %s: registering %s uids to be used." % (self.id, len(uids)))
        self._uids_to_use = uids

    def update_weights(self, uids: RecordsBatch, upd: DataTensor):
        X = self._dataset[self._data_params.train_split][self._data_params.features_key][[int(x) for x in uids]]
        self._model.update_weights(X, upd, collected_from_arbiter=True)


    def initialize(self):
        logger.info("Member %s: initializing" % self.id)
        self._dataset = self.processor.fit_transform()
        self._data_params = self.processor.data_params

        self._common_params = self.processor.common_params
        self.initialize_model()
        self.is_initialized = True
        logger.info("Member %s: has been initialized" % self.id)

    def finalize(self):
        pass

    def initialize_model(self):
        self._model = LogisticRegressionBatch(
            input_dim=self._dataset[self._data_params.train_split][self._data_params.features_key].shape[1],
            # output_dim=self._dataset[self._data_params.train_split][self._data_params.label_key].shape[1], # TODO
            output_dim=1,
            learning_rate=self._common_params.learning_rate,
            class_weights=None,
            init_weights=0.005
        )

