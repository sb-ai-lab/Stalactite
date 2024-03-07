import logging
from typing import Any, List, Optional

import mlflow
import numpy as np
import torch
from sklearn import metrics
from sklearn.metrics import roc_auc_score

from stalactite.base import Batcher, RecordsBatch, DataTensor, PartyDataTensor
from stalactite.batching import ListBatcher
from stalactite.helpers import log_timing
from stalactite.metrics import ComputeAccuracy_numpy
from stalactite.ml.arbitered.base import ArbiteredPartyMaster, SecurityProtocol, T, Role
from stalactite.ml.arbitered.logistic_regression.party_agent import ArbiteredPartyAgentLogReg

logger = logging.getLogger(__name__)


class ArbiteredPartyMasterLogReg(ArbiteredPartyAgentLogReg, ArbiteredPartyMaster):
    role: Role = Role.master

    def __init__(
            self,
            uid: str,
            epochs: int,
            num_classes: int,
            report_train_metrics_iteration: int,
            report_test_metrics_iteration: int,
            target_uids: List[str],
            inference_target_uids: List[str],
            batch_size: int,
            eval_batch_size: int,
            model_update_dim_size: int,
            security_protocol: Optional[SecurityProtocol] = None,
            l2_alpha: Optional[float] = None,
            do_train: bool = True,
            do_predict: bool = False,
            model_path: Optional[str] = None,
            do_save_model: bool = False,
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
        self.num_classes = num_classes
        self.report_train_metrics_iteration = report_train_metrics_iteration
        self.report_test_metrics_iteration = report_test_metrics_iteration
        self.l2_alpha = l2_alpha
        self.target_uids = target_uids
        self.inference_target_uids = inference_target_uids
        self.is_initialized = False
        self.is_finalized = False
        self._batch_size = batch_size
        self._eval_batch_size = eval_batch_size
        self._weights_dim = model_update_dim_size
        self.run_mlflow = run_mlflow
        self.processor = processor
        self.iteration_counter = 0
        self.party_predictions = dict()
        self.updates = dict()
        self.security_protocol = security_protocol
        self.do_train = do_train
        self.do_predict = do_predict
        self.do_save_model = do_save_model
        self.model_path = model_path

    def predict_partial(self, uids: RecordsBatch, batch: Any = None) -> DataTensor:
        logger.info("Master %s: predicting. Batch size: %s" % (self.id, len(uids)))
        self.check_if_ready()
        Xw, y = self.predict(uids, is_test=False, batch=batch)
        d = 0.25 * Xw - 0.5 * y.T.unsqueeze(2)
        return d

    def aggregate_partial_predictions(
            self,
            master_prediction: DataTensor,
            members_predictions: List[T],
            uids: RecordsBatch,
    ) -> T:
        class_predictions = []
        for class_idx in range(self.num_classes):
            prediction = master_prediction[class_idx]
            for member_preds in members_predictions:
                if self.security_protocol is not None:
                    prediction = self.security_protocol.add_matrices(prediction, member_preds[class_idx])
                else:
                    prediction += member_preds[class_idx]
            class_predictions.append(prediction)
        stacking_func = np.stack if self.security_protocol is not None else torch.stack
        master_prediction = stacking_func(class_predictions)

        return master_prediction

    def initialize(self, is_infer: bool = False):
        logger.info("Master %s: initializing" % self.id)
        dataset = self.processor.fit_transform()
        self._dataset = dataset

        self._data_params = self.processor.data_params
        self._common_params = self.processor.common_params

        self.target = dataset[self._data_params.train_split][self._data_params.label_key]
        self.test_target = dataset[self._data_params.test_split][self._data_params.label_key]

        assert self.target.shape[1] == self.num_classes, f'Inconsistent target shape with number of classes: ' \
                                                         f'`num_classes`: {self.num_classes}, `target`: ' \
                                                         f'{self.target.shape[1]}'

        # We need to change the labels from {0, 1} to {-1, 1} for the Taylor gradient approximation application
        self.target = torch.where(self.target == 0., -1., 1.)
        self.test_target = torch.where(self.test_target == 0., -1., 1.)

        self.initialize_model()
        self.is_initialized = True
        logger.info("Master %s: is initialized" % self.id)

    def predict(self, uids: Optional[List[str]], is_test: bool = False, batch: Any = None):
        split = self._data_params.train_split if not is_test else self._data_params.test_split
        target = self.target if not is_test else self.test_target
        batch_size = self._batch_size if not is_test else self._eval_batch_size
        if is_test and uids is None:
            X = self._dataset[split][self._data_params.features_key]
        else:
            if uids is None:
                uids = self._uids_to_use
            X = self._dataset[split][self._data_params.features_key][[int(x) for x in uids]]

        if batch is not None:
            # TODO: bugfix target double shuffle (remove dependency from batch number)
            y = target[batch_size * batch.subiter_seq_num: batch_size * (batch.subiter_seq_num + 1)]
        else:
            y = target

        return torch.stack([model.predict(X) for model in self._model]), y

    def aggregate_predictions(self, master_predictions: DataTensor, members_predictions: PartyDataTensor) -> DataTensor:
        logger.info(f"Master {self.id}: aggregates predictions from {len(members_predictions)} members")

        if master_predictions.shape[0] != len(self._model) or members_predictions[0].shape[0] != len(self._model):
            raise RuntimeError(
                f'Incorrect number of the predictions to aggregate (master predictions: '
                f'{master_predictions.shape[0]}), members_predictions: {members_predictions[0].shape[0]}, '
                f'number of models: {len(self._model)}'
            )

        predictions = []
        for class_idx in range(len(self._model)):
            predictions.append(
                torch.sigmoid(
                    torch.sum(
                        torch.hstack(
                            [master_predictions[class_idx]] +
                            [member_pred[class_idx] for member_pred in members_predictions]
                        ),
                        dim=1
                    )
                )
            )
        predictions = torch.stack(predictions).T

        return predictions

    def report_metrics(self, y: DataTensor, predictions: DataTensor, name: str, step: int):
        # TODO different metric for the inference
        y = torch.where(y == -1., -0., 1.)  # After a sigmoid function we calculate metrics on the {0, 1} labels
        y = y.numpy()

        predictions = predictions.detach().numpy()

        mae = metrics.mean_absolute_error(y, predictions)
        acc = ComputeAccuracy_numpy(is_linreg=False).compute(y, predictions)
        logger.info(f"Master %s: %s metrics (MAE): {mae}" % (self.id, name))
        logger.info(f"Master %s: %s metrics (Accuracy): {acc}" % (self.id, name))
        for avg in ["macro", "micro"]:
            try:
                roc_auc = roc_auc_score(y, predictions, average=avg)
            except ValueError:
                roc_auc = 0
            logger.info(f'{name} ROC AUC {avg}: {roc_auc}')
            if self.run_mlflow:
                mlflow.log_metric(f"{name.lower()}_roc_auc_{avg}", roc_auc, step=step)

    def make_batcher(
            self,
            uids: Optional[List[str]] = None,
            party_members: Optional[List[str]] = None,
            is_infer: bool = False
    ) -> Batcher:
        logger.info("Master %s: making a make_batcher for uids %s" % (self.id, len(uids)))
        self.check_if_ready()
        batch_size = self._eval_batch_size if is_infer else self._batch_size
        epochs = 1 if is_infer else self.epochs
        if uids is None:
            raise RuntimeError('Master must initialize batcher with collected uids.')
        return ListBatcher(epochs=epochs, members=party_members, uids=uids, batch_size=batch_size)
