from abc import ABC
from copy import copy
from typing import List, Optional, Any, Union, Tuple
import logging

import datasets
import numpy as np
import torch
from pydantic import BaseModel

from stalactite.base import RecordsBatch, DataTensor, PartyAgent
from stalactite.ml.arbitered.base import SecurityProtocol, T
from stalactite.utils import Role
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
    _uids_to_use_test: Optional[RecordsBatch]

    _dataset: datasets.DatasetDict
    _data_params: BaseModel
    _common_params: BaseModel

    security_protocol: SecurityProtocol

    epochs: int
    l2_alpha: Optional[float]
    _batch_size: int
    _eval_batch_size: int
    num_classes: int
    use_inner_join: bool
    device: torch.device

    device_dataset_train_split: torch.Tensor
    device_dataset_test_split: torch.Tensor

    ovr = True

    def prepare_device_data(self, is_infer: bool = False):
        if not is_infer:
            self.device_dataset_train_split = self._dataset[self._data_params.train_split][
                self._data_params.features_key].to(self.device)
        else:
            self.device_dataset_test_split = self._dataset[self._data_params.test_split][
                self._data_params.features_key].to(self.device)

    def initialize_model_from_params(self, **model_params) -> Any:
        return LogisticRegressionBatch(**model_params).to(device=self.device)

    def update_weights(self, uids: RecordsBatch, upd: DataTensor):
        logger.info(f"{self.role.capitalize()} {self.id}: updating weights. Incoming tensor: {upd.size()}")
        tensor_idx = [self.uid2tensor_idx[uid] for uid in uids] if uids else None
        X = self.device_dataset_train_split[tensor_idx, :]
        if upd.shape[0] != len(self._model):
            raise RuntimeError(
                f'Incorrect number of the updates were received (number of models to update: {upd.shape[0]}), '
                f'number of models: {len(self._model)}'
            )
        for upd_i, model in zip(upd, self._model):
            model.update_weights(X, upd_i.to(device=self.device), collected_from_arbiter=True)
        logger.info(f"{self.role.capitalize()} {self.id}: successfully updated weights")

    def compute_gradient(self, aggregated_predictions_diff: T, uids: List[str]) -> T:
        logger.info(f'{self.role.capitalize()} {self.id} computes gradient')
        tensor_idx = [self.uid2tensor_idx[uid] for uid in uids] if uids else None
        X = self.device_dataset_train_split[tensor_idx, :]
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
            g = torch.stack([torch.matmul(x, pred.to(device=self.device)) for pred in aggregated_predictions_diff])
            if self.l2_alpha is not None:
                weights_sum = [model.get_weights().T.to(device=self.device) for model in self._model]
                g = torch.stack([self.l2_alpha * w_sum + g_class for w_sum, g_class in zip(weights_sum, g)])
        logger.debug(f'{self.role.capitalize()} {self.id} successfully computed gradient')
        return g

    def records_uids(self, is_infer: bool = False) -> Union[List[str], Tuple[List[str], bool]]:
        logger.info(f"{self.role.capitalize()} {self.id}: reporting existing record uids")
        if not is_infer:
            if self.role == Role.master:
                return self.target_uids
            elif self.role == Role.member:
                return self._uids, self.use_inner_join
        else:
            if self.role == Role.master:
                return self.inference_target_uids
            elif self.role == Role.member:
                return self._infer_uids, self.use_inner_join

    def register_records_uids(self, uids: List[str], is_infer: bool = False):
        logger.info(f"{self.role.capitalize()} {self.id}: registering {len(uids)} uids to be used.")
        if is_infer:
            self._uids_to_use_test = uids
        else:
            self._uids_to_use = uids
        if self.role == Role.member:
            self.fillna(is_infer=is_infer)

    def initialize_model(self, do_load_model: bool = False):
        logger.info(f"{self.role.capitalize()} {self.id} initializes model on device: {self.device}")
        logger.debug(f"Model is loaded from path: {do_load_model}")
        if do_load_model:
            self._model = self.load_model(is_ovr_models=self.ovr)
        else:
            input_dim = self.device_dataset_train_split.shape[1]
            model_type = 'OVR models' if self.num_classes > 1 else 'binary model'
            logger.info(f'{self.id} Initializing {model_type} for {self.num_classes} classes')
            self._model = [
                LogisticRegressionBatch(
                    input_dim=input_dim,
                    output_dim=1,
                    init_weights=0.005
                ) for _ in range(self.num_classes)
            ]
        self._model = [model.to(device=self.device) for model in self._model]

    def finalize(self, is_infer: bool = False):
        logger.info(f"{self.role.capitalize()} {self.id}: finalizing")
        self.check_if_ready()
        if self.do_save_model and not is_infer:
            self.save_model(is_ovr_models=self.ovr)
        self.is_finalized = True
        logger.info(f"{self.role.capitalize()} {self.id} has finalized")

    def fillna(self, is_infer: bool = False) -> None:
        """ Fills missing values for member's dataset"""
        uids_to_use = self._uids_to_use_test if is_infer else self._uids_to_use
        _uids = self._infer_uids if is_infer else self._uids
        _uid2tensor_idx = self.uid2tensor_idx_test if is_infer else self.uid2tensor_idx
        split = self._data_params.test_split if is_infer else self._data_params.train_split

        uids_to_fill = list(set(uids_to_use) - set(_uids))
        if len(uids_to_fill) == 0:
            return

        logger.info(f"{self.role.capitalize()} {self.id} has {len(uids_to_fill)} missing values: using fillna...")
        start_idx = max(_uid2tensor_idx.values()) + 1
        idx = start_idx
        for uid in uids_to_fill:
            _uid2tensor_idx[uid] = idx
            idx += 1

        fill_shape = self.device_dataset_test_split.shape[1] if is_infer else self.device_dataset_train_split.shape[1]
        member_id = int(self.id.split("-")[-1]) + 1
        features = copy(self.device_dataset_test_split.cpu()) if is_infer else copy(self.device_dataset_train_split.cpu())
        features = torch.cat([features, torch.zeros((len(uids_to_fill), fill_shape))])
        has_features_column = torch.tensor([1.0 for _ in range(start_idx)] + [0.0 for _ in range(len(uids_to_fill))])
        features = torch.cat([features, torch.unsqueeze(has_features_column, 1)], dim=1)

        ds = datasets.Dataset.from_dict(
            {
                "user_id": list(_uid2tensor_idx.keys()),
                f"features_part_{member_id}": features,
            }
        )

        ds = ds.with_format("torch")
        self._dataset[split] = ds
        self.prepare_device_data(is_infer=is_infer)

        if not is_infer:
            self.initialize_model()
