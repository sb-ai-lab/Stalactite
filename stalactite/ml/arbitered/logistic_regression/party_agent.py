from abc import ABC
from typing import List, Optional, Any, Union
import logging

import datasets
import numpy as np
import torch
from pydantic import BaseModel

from stalactite.base import RecordsBatch, DataTensor, PartyAgent
from stalactite.helpers import log_timing
from stalactite.ml.arbitered.base import SecurityProtocol, T, Role
from stalactite.models import LogisticRegressionBatch

logger = logging.getLogger(__name__)


class ArbiteredPartyAgentLogReg(PartyAgent, ABC):
    id: str
    role: Role

    _model: Union[LogisticRegressionBatch, List[LogisticRegressionBatch]]

    _uids: Optional[RecordsBatch]
    _infer_uids: Optional[RecordsBatch]
    target_uids: Optional[RecordsBatch]
    inference_target_uids: Optional[RecordsBatch]
    _uids_to_use: Optional[RecordsBatch]

    _dataset: datasets.DatasetDict
    _data_params: BaseModel
    _common_params: BaseModel

    security_protocol: SecurityProtocol

    epochs: int
    l2_alpha: Optional[float]
    _batch_size: int
    _eval_batch_size: int
    num_classes: int

    def initialize_model_from_params(self, **model_params) -> Any:
        return LogisticRegressionBatch(**model_params)

    def update_weights(self, uids: RecordsBatch, upd: DataTensor):
        X = self._dataset[self._data_params.train_split][self._data_params.features_key][[int(x) for x in uids]]
        if upd.shape[0] != len(self._model):
            raise RuntimeError(
                f'Incorrect number of the updates were received (number of models to update: {upd.shape[0]}), '
                f'number of models: {len(self._model)}'
            )
        for upd_i, model in zip(upd, self._model):
            model.update_weights(X, upd_i, collected_from_arbiter=True)

    def compute_gradient(self, aggregated_predictions_diff: T, uids: List[str]) -> T:
        with log_timing(f'{self.id} compute gradient.'):
            X = self._dataset[self._data_params.train_split][self._data_params.features_key][[int(x) for x in uids]]
            if self.security_protocol is not None:
                x = self.security_protocol.encode(X.T / X.shape[0])
                g = np.stack([
                        self.security_protocol.multiply_plain_cypher(x, pred)
                        for pred in aggregated_predictions_diff
                    ])
                if self.l2_alpha is not None:
                    weights_sum = [model.get_weights().T for model in self._model]
                    g = np.stack([
                        self.security_protocol.add_matrices(self.l2_alpha * w_sum, g_class)
                        for w_sum, g_class in zip(weights_sum, g)
                    ])
            else:
                x = X.T / X.shape[0]
                g = torch.stack([torch.matmul(x, pred) for pred in aggregated_predictions_diff])
                if self.l2_alpha is not None:
                    weights_sum = [model.get_weights().T for model in self._model]
                    g = torch.stack([self.l2_alpha * w_sum + g_class for w_sum, g_class in zip(weights_sum, g)])
            return g

    def records_uids(self, is_infer: bool = False) -> List[str]:
        if not is_infer:
            if self.role == Role.master:
                return self.target_uids
            elif self.role == Role.member:
                return self._uids
        else:
            if self.role == Role.master:
                return self.inference_target_uids
            elif self.role == Role.member:
                return self._infer_uids

    def register_records_uids(self, uids: List[str]):
        logger.info("%s: registering %s uids to be used." % (self.id, len(uids)))
        self._uids_to_use = uids

    def initialize_model(self):
        input_dim = self._dataset[self._data_params.train_split][self._data_params.features_key].shape[1]
        model_type = 'OVR models' if self.num_classes > 1 else 'binary model'
        logger.info(f'{self.id} Initializing {model_type} for {self.num_classes} classes')
        self._model = [
            LogisticRegressionBatch(
                input_dim=input_dim,
                output_dim=1,
                init_weights=0.005
            ) for _ in range(self.num_classes)
        ]

    def finalize(self, is_infer: bool = False):
        self.check_if_ready()
        if self.do_save_model and not is_infer:
            self.save_model()
        self.is_finalized = True
        logger.info(f"{self.role.capitalize()} {self.id} has finalized")
