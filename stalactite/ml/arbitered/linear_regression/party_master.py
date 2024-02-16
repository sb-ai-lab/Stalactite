import logging
from typing import Any, List

import datasets
from pydantic import BaseModel

from stalactite.base import Batcher, PartyCommunicator, RecordsBatch, DataTensor, PartyDataTensor
from stalactite.batching import ListBatcher
from stalactite.configs import VFLModelConfig
from stalactite.data_preprocessors import ImagePreprocessor
from stalactite.ml.arbitered.base import ArbiteredPartyMaster
from stalactite.models import LinearRegressionBatch

logger = logging.getLogger(__name__)

logger.setLevel(logging.DEBUG)
sh = logging.StreamHandler()
sh.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
logger.addHandler(sh)


class ArbiteredPartyMasterLinReg(ArbiteredPartyMaster):
    _data_params: BaseModel
    _dataset: datasets.DatasetDict
    _model: LinearRegressionBatch
    _common_params: VFLModelConfig
    _uids_to_use: List[str]

    def __init__(
            self,
            uid: str,
            epochs: int,
            report_train_metrics_iteration: int,
            report_test_metrics_iteration: int,
            target_uids: List[str],
            batch_size: int,
            model_update_dim_size: int,
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

    def update_weights(self, uids: RecordsBatch, upd: DataTensor):
        # TODO will be a dupe
        logger.info("Master %s: updating weights. Incoming tensor: %s" % (self.id, tuple(upd.size())))
        self._check_if_ready()
        X_train = self._dataset[self._data_params.train_split][self._data_params.features_key][[int(x) for x in uids]]
        self._model.update_weights(X_train, upd)
        logger.info("Master %s: successfully updated weights" % self.id)

    def compute_gradient(self, aggregated_predictions_diff: DataTensor, uids: List[str]) -> DataTensor:
        pass # TODO >>>???????????

    def make_batcher(self, uids: List[str], party_members: List[str]) -> Batcher:
        logger.info("Master %s: making a batcher for uids %s" % (self.id, uids))
        self._check_if_ready()
        assert party_members is not None, "Master is trying to initialize batcher without members list"
        return ListBatcher(epochs=self.epochs, members=party_members, uids=uids, batch_size=self._batch_size)

    def records_uids(self) -> List[str]:
        return self.target_uids

    def register_records_uids(self, uids: List[str]):
        self._uids_to_use = uids

    def initialize_model(self):
        self._model = LinearRegressionBatch(
            input_dim=self._dataset[self._data_params.train_split][self._data_params.features_key].shape[1],
            output_dim=1, reg_lambda=0.5
        )

    def initialize(self):
        logger.info("Master %s: initializing" % self.id)
        dataset = self.processor.fit_transform()
        self._dataset = dataset

        # self.target = dataset[self.processor.data_params.train_split][self.processor.data_params.label_key]
        # self.test_target = dataset[self.processor.data_params.test_split][self.processor.data_params.label_key]
        #
        self._data_params = self.processor.data_params
        self._common_params = self.processor.common_params

        self.initialize_model()
        self.is_initialized = True
        logger.info("Master %s: is initialized" % self.id)

    def finalize(self):
        self._check_if_ready()
        self.is_finalized = True
        logger.info("Master %s: has finalized" % self.id)

    def _check_if_ready(self):
        if not self.is_initialized and not self.is_finalized:
            raise RuntimeError("The member has not been initialized")

    def predict_partial(self, uids: RecordsBatch, subiter_seq_num: int) -> DataTensor:
        logger.info("Master %s: predicting. Batch size: %s" % (self.id, len(uids)))
        self._check_if_ready()
        y = self.target[self._batch_size * subiter_seq_num: self._batch_size * (subiter_seq_num + 1)]
        X = self._dataset[self._data_params.train_split][self._data_params.features_key][[int(x) for x in uids]]
        predictions = self._model.predict(X)
        logger.info("Master %s: made predictions." % self.id)
        return predictions - y

    def aggregate_partial_predictions(
            self, master_prediction: DataTensor, members_predictions: PartyDataTensor
    ) -> DataTensor:
        for member_pred in members_predictions:
            master_prediction += member_pred
        return master_prediction
