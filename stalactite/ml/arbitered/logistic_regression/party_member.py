from typing import List, Optional, Any
import logging

import numpy as np
import torch

from stalactite.base import RecordsBatch, DataTensor, Batcher, PartyCommunicator
from stalactite.batching import ListBatcher
from stalactite.helpers import log_timing
from stalactite.ml.arbitered.base import ArbiteredPartyMember, SecurityProtocol
from stalactite.models import LogisticRegressionBatch

logger = logging.getLogger(__name__)

class ArbiteredPartyMemberLogReg(ArbiteredPartyMember):
    def inference_loop(self, batcher: Batcher, party: PartyCommunicator) -> None:
        pass

    def initialize_model_from_params(self, **model_params) -> Any:
        pass

    def __init__(
            self,
            uid: str,
            epochs: int,
            batch_size: int,
            eval_batch_size: int,
            member_record_uids: List[str],
            security_protocol: SecurityProtocol,
            processor=None,
    ) -> None:
        self.id = uid
        self.epochs = epochs
        self._batch_size = batch_size
        self._eval_batch_size = eval_batch_size
        self._uids = member_record_uids
        self.processor = processor
        self.security_protocol = security_protocol
        self._batcher = None

    _model: Optional[LogisticRegressionBatch] = None

    def predict_partial(self, uids: RecordsBatch) -> DataTensor:
        # X = self._dataset[self._data_params.train_split][self._data_params.features_key][[int(x) for x in uids]]
        predictions = self.predict(uids, is_test=False)
        Xw = 0.25 * predictions
        Xw = self.security_protocol.encrypt(Xw)
        return Xw

    def predict(self, uids: Optional[List[str]], is_test: bool = False) -> DataTensor:
        with log_timing(f'Prediction on the member {self.id}', log_func=print):
            if not is_test:
                if uids is None:
                    uids = self._uids_to_use
                X = self._dataset[self._data_params.train_split][self._data_params.features_key][[int(x) for x in uids]]
            else:
                X = self._dataset[self._data_params.test_split][self._data_params.features_key]
            print('predict', X.shape, X.dtype)
            return self._model.predict(X)

    def compute_gradient(self, aggregated_predictions_diff: DataTensor, uids: List[str]) -> DataTensor:
        with log_timing(f'{self.id} compute gradient.'):
            X = self._dataset[self._data_params.train_split][self._data_params.features_key][[int(x) for x in uids]]
            # g = torch.matmul(X.T, aggregated_predictions_diff) / X.shape[0]
            # Xt = X.T.numpy(force=True).astype('float')
            # print(self.id, 'X.t', Xt.shape)
            # g = np.dot(Xt, aggregated_predictions_diff) / X.shape[0]

            g = self.security_protocol.multiply_plain_cypher(X.T, aggregated_predictions_diff)
            print(self.id, 'g.shape', g.shape, g[0], )
            return g

    # def make_batcher(self) -> Batcher:
    def make_batcher(
            self,
            uids: Optional[List[str]] = None,
            party_members: Optional[List[str]] = None,
            is_infer: bool = False
    ) -> Batcher:
        if uids is None:
            uids = self._uids_to_use
        epochs = 0 if is_infer else self.epochs
        batch_size = self._eval_batch_size if is_infer else self._batch_size
        batcher = ListBatcher(
            epochs=epochs,
            members=None,
            uids=uids,
            batch_size=batch_size
        )
        self._batcher = batcher
        return batcher

    def records_uids(self, is_infer: bool = False) -> List[str]:
        logger.info("Member %s: reporting existing record uids" % self.id)
        return self._uids

    def register_records_uids(self, uids: List[str]):
        logger.info("Member %s: registering %s uids to be used." % (self.id, len(uids)))
        self._uids_to_use = uids

    def update_weights(self, uids: RecordsBatch, upd: DataTensor):
        X = self._dataset[self._data_params.train_split][self._data_params.features_key][[int(x) for x in uids]]
        self._model.update_weights(X, upd, collected_from_arbiter=True)


    def initialize(self, is_infer: bool = False):
        logger.info("Member %s: initializing" % self.id)
        self._dataset = self.processor.fit_transform()
        self._data_params = self.processor.data_params

        self._common_params = self.processor.common_params
        self.initialize_model()
        self.is_initialized = True
        logger.info("Member %s: has been initialized" % self.id)

    def finalize(self, is_infer: bool = False):
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

