from typing import List, Optional, Union
import logging

import torch

from stalactite.base import RecordsBatch, DataTensor, Batcher
from stalactite.batching import ListBatcher
from stalactite.ml.arbitered.base import ArbiteredPartyMember, SecurityProtocol
from stalactite.utils import Role
from stalactite.ml.arbitered.logistic_regression.party_agent import ArbiteredPartyAgentLogReg

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
            use_inner_join: bool = True,
            seed: int = None,
            device: str = 'cpu',
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
        self.use_inner_join = use_inner_join
        self.seed = seed
        self.device = torch.device(device)

    def predict_partial(self, uids: RecordsBatch) -> DataTensor:
        logger.info(f'Member {self.id} makes partial predictions')
        predictions = self.predict(uids, is_infer=False)
        Xw = 0.25 * predictions
        if self.security_protocol is not None:
            Xw = self.security_protocol.encrypt(Xw)
        logger.debug(f"Member {self.id}: made partial predictions")
        return Xw

    def predict(self, uids: Optional[List[str]], is_infer: bool = False) -> Union[DataTensor, List[DataTensor]]:
        logger.info(f"Member {self.id}: predicting")
        _uid2tensor_idx = self.uid2tensor_idx_test if is_infer else self.uid2tensor_idx
        if uids is None:
            uids = self._uids_to_use_test if is_infer else self._uids_to_use
        tensor_idx = [_uid2tensor_idx[uid] for uid in uids]
        X = self.device_dataset_train_split[tensor_idx] if not is_infer else self.device_dataset_test_split[tensor_idx]
        logger.debug(f"Member {self.id}: made predictions")
        return torch.stack([model.predict(X) for model in self._model])

    def make_batcher(
            self,
            uids: Optional[List[str]] = None,
            party_members: Optional[List[str]] = None,
            is_infer: bool = False
    ) -> Batcher:
        if uids is None:
            uids = self._uids_to_use_test if is_infer else self._uids_to_use
        logger.info(f"Member {self.id} makes a batcher for {len(uids)} uids")
        epochs = 1 if is_infer else self.epochs
        batch_size = self._eval_batch_size if is_infer else self._batch_size
        return ListBatcher(epochs=epochs, members=None, uids=uids, batch_size=batch_size)

    def initialize(self, is_infer: bool = False):
        logger.info(f"Member {self.id}: initializing")
        self._dataset = self.processor.fit_transform()

        self._data_params = self.processor.data_params

        self.prepare_device_data(is_infer=False)
        self.prepare_device_data(is_infer=True)

        self._common_params = self.processor.common_params
        self.uid2tensor_idx = {uid: i for i, uid in enumerate(self._uids)}
        self.uid2tensor_idx_test = {uid: i for i, uid in enumerate(self._infer_uids)}

        self.initialize_model(do_load_model=is_infer)

        self.is_initialized = True
        self.is_finalized = False
        logger.info(f"Member {self.id}: has been initialized")
