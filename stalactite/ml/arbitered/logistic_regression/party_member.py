from typing import List, Optional, Any, Union
import logging

import numpy as np
import torch

from stalactite.base import RecordsBatch, DataTensor, Batcher, PartyCommunicator
from stalactite.batching import ListBatcher
from stalactite.helpers import log_timing
from stalactite.ml.arbitered.base import ArbiteredPartyMember, SecurityProtocol, T, Role
from stalactite.ml.arbitered.logistic_regression.party_agent import ArbiteredPartyAgentLogReg
from stalactite.models import LogisticRegressionBatch

logger = logging.getLogger(__name__)


class ArbiteredPartyMemberLogReg(ArbiteredPartyAgentLogReg, ArbiteredPartyMember):
    role: Role = Role.member

    def __init__(
            self,
            uid: str,
            epochs: int,
            batch_size: int,
            num_classes: int,
            report_train_metrics_iteration: int,
            report_test_metrics_iteration: int,
            eval_batch_size: int,
            member_record_uids: List[str],
            member_inference_record_uids: List[str],
            security_protocol: Optional[SecurityProtocol] = None,
            l2_alpha: Optional[float] = None,
            do_train: bool = True,
            do_predict: bool = False,
            model_path: Optional[str] = None,
            do_save_model: bool = False,
            processor=None,
    ) -> None:
        self.id = uid
        self.epochs = epochs
        self.num_classes = num_classes
        self._batch_size = batch_size
        self._eval_batch_size = eval_batch_size
        self._uids = member_record_uids
        self._infer_uids = member_inference_record_uids
        self.processor = processor
        self.l2_alpha = l2_alpha
        self.security_protocol = security_protocol
        self._batcher = None
        self.do_train = do_train
        self.do_predict = do_predict
        self.report_train_metrics_iteration = report_train_metrics_iteration
        self.report_test_metrics_iteration = report_test_metrics_iteration
        self.do_save_model = do_save_model
        self.model_path = model_path

        self.ovr = None

    def predict_partial(self, uids: RecordsBatch) -> DataTensor:
        predictions = self.predict(uids, is_test=False)
        Xw = 0.25 * predictions
        if self.security_protocol is not None:
            Xw = self.security_protocol.encrypt(Xw)
        return Xw

    def predict(self, uids: Optional[List[str]], is_test: bool = False) -> Union[DataTensor, List[DataTensor]]:
        with log_timing(f'Prediction on the member {self.id}', log_func=print):
            split = self._data_params.train_split if not is_test else self._data_params.test_split
            if is_test and uids is None:
                X = self._dataset[split][self._data_params.features_key]
            else:
                if uids is None:
                    uids = self._uids_to_use
                X = self._dataset[split][self._data_params.features_key][[int(x) for x in uids]]
            return torch.stack([model.predict(X) for model in self._model])

    def make_batcher(
            self,
            uids: Optional[List[str]] = None,
            party_members: Optional[List[str]] = None,
            is_infer: bool = False
    ) -> Batcher:
        if uids is None:
            uids = self._uids_to_use
        epochs = 1 if is_infer else self.epochs
        batch_size = self._eval_batch_size if is_infer else self._batch_size
        return ListBatcher(epochs=epochs, members=None, uids=uids, batch_size=batch_size)

    def initialize(self, is_infer: bool = False):
        logger.info("Member %s: initializing" % self.id)
        self._dataset = self.processor.fit_transform()
        self._data_params = self.processor.data_params
        self._common_params = self.processor.common_params

        self.initialize_model()

        self.is_initialized = True
        logger.info("Member %s: has been initialized" % self.id)


