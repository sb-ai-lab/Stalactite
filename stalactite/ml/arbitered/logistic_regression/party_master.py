import logging
from typing import Any, List, Optional

import datasets
import numpy as np
import torch
from pydantic import BaseModel
from sklearn import metrics
from sklearn.metrics import roc_auc_score

from stalactite.base import Batcher, PartyCommunicator, RecordsBatch, DataTensor, PartyDataTensor
from stalactite.batching import ListBatcher
from stalactite.configs import VFLModelConfig
from stalactite.data_preprocessors import ImagePreprocessor
from stalactite.metrics import ComputeAccuracy_numpy
from stalactite.ml.arbitered.base import ArbiteredPartyMaster, SecurityProtocol
from stalactite.models import LogisticRegressionBatch

logger = logging.getLogger(__name__)

logger.setLevel(logging.DEBUG)
sh = logging.StreamHandler()
sh.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
logger.addHandler(sh)


class ArbiteredPartyMasterLogReg(ArbiteredPartyMaster):
    _data_params: BaseModel
    _dataset: datasets.DatasetDict
    _model: LogisticRegressionBatch
    _common_params: VFLModelConfig
    _uids_to_use: List[str]
    _pos_weight: float = 1

    def __init__(
            self,
            uid: str,
            epochs: int,
            report_train_metrics_iteration: int,
            report_test_metrics_iteration: int,
            target_uids: List[str],
            batch_size: int,
            model_update_dim_size: int,
            security_protocol: SecurityProtocol,
            processor=None,
            run_mlflow: bool = False,
    ) -> None:
        """ Initialize ArbiteredPartyMasterLinReg.

        :param uid: Unique identifier for the party master.
        :param epochs: Number of training epochs.
        :param report_train_metrics_iteration: Number of iterations between reporting metrics on the train dataset.
        :param report_test_metrics_iteration: Number of iterations between reporting metrics on the test dataset.
        :param target_uids: List of unique identifiers for target dataset rows.
        :param batch_size: Size of the training batch.
        :param model_update_dim_size: Dimension size for model updates.
        :param processor: Optional data processor.
        :param run_mlflow: Flag indicating whether to use MlFlow for logging.

        :return: None
        """
        self.id = uid
        self.epochs = epochs
        self.report_train_metrics_iteration = report_train_metrics_iteration
        self.report_test_metrics_iteration = report_test_metrics_iteration
        self.target_uids = target_uids
        self.is_initialized = False
        self.is_finalized = False
        self._batch_size = batch_size
        self._weights_dim = model_update_dim_size
        self.run_mlflow = run_mlflow
        self.processor = processor
        self.iteration_counter = 0
        self.party_predictions = dict()
        self.updates = dict()
        self.security_protocol = security_protocol

    def update_weights(self, uids: RecordsBatch, upd: DataTensor):
        X = self._dataset[self._data_params.train_split][self._data_params.features_key][[int(x) for x in uids]]
        self._model.update_weights(X, upd, collected_from_arbiter=True)

    def predict_partial(self, uids: RecordsBatch) -> DataTensor:
        logger.info("Master %s: predicting. Batch size: %s" % (self.id, len(uids)))
        self.check_if_ready()
        # X = self._dataset[self._data_params.train_split][self._data_params.features_key][[int(x) for x in uids]]
        # Xw = self._model.predict(X)
        Xw, y = self.predict(uids, is_test=False)

        d = 0.25 * Xw - 0.5 * y
        return d

    def aggregate_partial_predictions(
            self, master_prediction: DataTensor, members_predictions: PartyDataTensor, uids: RecordsBatch,
    ) -> DataTensor:
        y = self.target[[int(x) for x in uids]]
        for member_pred in members_predictions:
            master_prediction += member_pred

        # weights = torch.where(y == 1, self._pos_weight, 1)
        # return master_prediction * weights
        return master_prediction


    def compute_gradient(self, aggregated_predictions_diff: DataTensor, uids: List[str]) -> DataTensor:
        X = self._dataset[self._data_params.train_split][self._data_params.features_key][[int(x) for x in uids]]
        g = torch.matmul(X.T, aggregated_predictions_diff) / X.shape[0]
        return g

    def records_uids(self) -> List[str]:
        return self.target_uids

    def register_records_uids(self, uids: List[str]):
        self._uids_to_use = uids

    def initialize_model(self):
        self._model = LogisticRegressionBatch(
            input_dim=self._dataset[self._data_params.train_split][self._data_params.features_key].shape[1],
            # output_dim=self._dataset[self._data_params.train_split][self._data_params.label_key].shape[1], # TODO
            output_dim=1,
            learning_rate=self._common_params.learning_rate,
            class_weights=None,
            init_weights=0.005
        )

    def initialize(self):
        logger.info("Master %s: initializing" % self.id)
        dataset = self.processor.fit_transform()
        self._dataset = dataset

        self.target = dataset[self.processor.data_params.train_split][self.processor.data_params.label_key][:, 6].unsqueeze(1)
        self.test_target = dataset[self.processor.data_params.test_split][self.processor.data_params.label_key][:, 6].unsqueeze(1)

        unique, counts = np.unique(self.target, return_counts=True)
        self._pos_weight = counts[0] / counts[1]

        self._data_params = self.processor.data_params
        self._common_params = self.processor.common_params

        self.initialize_model()
        self.is_initialized = True
        logger.info("Master %s: is initialized" % self.id)

    def finalize(self):
        self.check_if_ready()
        self.is_finalized = True
        logger.info("Master %s: has finalized" % self.id)


    def predict(self, uids: Optional[List[str]], is_test: bool = False):
        if not is_test:
            if uids is None:
                uids = self._uids_to_use
            X = self._dataset[self._data_params.train_split][self._data_params.features_key][[int(x) for x in uids]]
            y = self.target[[int(x) for x in uids]]
        else:
            X = self._dataset[self._data_params.test_split][self._data_params.features_key]
            y = self.test_target
        Xw = self._model.predict(X)

        return Xw, y

    def aggregate_predictions(self, master_predictions: DataTensor, members_predictions: PartyDataTensor) -> DataTensor:
        predictions = torch.sigmoid(torch.sum(torch.hstack([master_predictions] + members_predictions), dim=1))
        return predictions

    def report_metrics(self, y: DataTensor, predictions: DataTensor, name: str):
        y = y.numpy()
        predictions = predictions.detach().numpy()

        mae = metrics.mean_absolute_error(y, predictions)
        acc = ComputeAccuracy_numpy(is_linreg=False).compute(y, predictions)
        print(f"Master %s: %s metrics (MAE): {mae}" % (self.id, name))
        print(f"Master %s: %s metrics (Accuracy): {acc}" % (self.id, name))
        for avg in ["macro", "micro"]:
            try:
                roc_auc = roc_auc_score(y, predictions, average=avg)
            except ValueError:
                roc_auc = 0
            print(f'{name} ROC AUC {avg}: {roc_auc}')


    @property
    def batcher(self) -> Batcher:
        return self._batcher
