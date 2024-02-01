from abc import abstractmethod
from typing import List, Optional
import logging

import mlflow
import torch
import scipy as sp
from datasets.dataset_dict import DatasetDict
from sklearn.metrics import mean_absolute_error, roc_auc_score, precision_recall_curve, auc
from sklearn.linear_model import LogisticRegression as LogRegSklearn

from stalactite.base import PartyMaster, DataTensor, PartyDataTensor
from stalactite.batching import Batcher, ListBatcher
from stalactite.metrics import ComputeAccuracy_numpy
from stalactite.models.linreg_batch import LinearRegressionBatch
from stalactite.models.logreg_batch import LogisticRegressionBatch

logger = logging.getLogger(__name__)

logger.setLevel(logging.DEBUG)
sh = logging.StreamHandler()
sh.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
logger.addHandler(sh)


class PartySingle:
    model_name: str
    member_id = 0

    def __init__(
            self,
            epochs: int,
            batch_size: int,
            processor=None,
            use_mlflow: bool = False,
            report_train_metrics_iteration: int = 1,
            report_test_metrics_iteration: int = 1,
            learning_rate: float = 0.01,

    ):
        self.epochs = epochs
        self.batch_size = batch_size
        self.use_mlflow = use_mlflow
        self.report_train_metrics_iteration = report_train_metrics_iteration
        self.report_test_metrics_iteration = report_test_metrics_iteration
        self.learning_rate = learning_rate
        self.processor = processor
        self._model = None
        self.is_initialized = False
        self.is_finalized = False

    def run(self):
        self.initialize()
        uids = self.synchronize_uids()
        self.loop(batcher=self.make_batcher(uids=uids))
        self.finalize()

    def loop(self, batcher: Batcher):
        for titer in batcher:
            step = titer.seq_num
            logger.debug(f"batch: {step}")
            batch = titer.batch
            tensor_idx = [int(x) for x in batch]

            x = self.x_train[tensor_idx]
            y = self.target[tensor_idx]

            self.update_weights(x, y)

            predictions = self.compute_predictions(is_test=False)
            predictions_test = self.compute_predictions(is_test=True)

            if (titer.previous_batch is None) and (step > 0):
                break
            step += 1

            if self.report_train_metrics_iteration > 0 and titer.seq_num % self.report_train_metrics_iteration == 0:
                self.report_metrics(self.target, predictions, name="Train", step=step)
            if self.report_test_metrics_iteration > 0 and titer.seq_num % self.report_test_metrics_iteration == 0:
                self.report_metrics(self.test_target, predictions_test, name="Test", step=step)

    def synchronize_uids(self) -> List[str]:
        return [str(x) for x in range(self.target.shape[0])]

    def make_batcher(self, uids: List[str]) -> Batcher:
        return ListBatcher(
            epochs=self.epochs,
            members=[str(x) for x in range(10)],
            uids=uids,
            batch_size=self.batch_size,
            shuffle=False
        )

    @abstractmethod
    def update_weights(
            self,
            x: DataTensor,
            y: DataTensor,
    ):
        ...

    @abstractmethod
    def compute_predictions(
            self,
            is_test: bool = False,
    ) -> DataTensor:
        ...

    @abstractmethod
    def initialize_model(self):
        ...

    def initialize(self):
        logger.info("Centralized experiment initializing")

        ds = self.processor.fit_transform()
        self.target = ds[self.processor.data_params.train_split][self.processor.data_params.label_key]
        self.test_target = ds[self.processor.data_params.test_split][self.processor.data_params.label_key]

        self.x_train = ds[self.processor.data_params.train_split][self.processor.data_params.features_key]
        self.x_test = ds[self.processor.data_params.test_split][self.processor.data_params.features_key]

        self.class_weights = self.processor.get_class_weights() \
            if self.processor.common_params.use_class_weights else None

        self._dataset = self.processor.fit_transform()
        self._data_params = self.processor.data_params
        self._common_params = self.processor.common_params
        self.initialize_model()
        self.is_initialized = True
        logger.info("Centralized experiment is initialized")

    def finalize(self):
        self.is_finalized = True
        logger.info("Experiment has finished")

    def report_metrics(self, y: DataTensor, predictions: DataTensor, name: str, step: int):
        acc = ComputeAccuracy_numpy(is_linreg=self.model_name == "linreg", is_multilabel=self.is_multilabel)

        mae = mean_absolute_error(y.numpy(), predictions)
        acc = acc.compute(true_label=y.numpy(), predictions=predictions)

        if self.use_mlflow:
            mlflow.log_metric(f"{name.lower()}_mae", mae, step=step)
            mlflow.log_metric(f"{name.lower()}_acc", acc, step=step)
        else:
            logger.info(f'{name} MAE on step {step}: {mae}')
            logger.info(f'{name} Accuracy on step {step}: {acc}')

        if self.n_labels > 1:
            for avg in ["macro", "micro"]:
                roc_auc = roc_auc_score(y.numpy(), predictions, average=avg)
                if self.use_mlflow:
                    mlflow.log_metric(f"{name.lower()}_roc_auc_{avg}", roc_auc, step=step)
                else:
                    logger.info(f'{name} roc_auc_{avg} on step {step}: {roc_auc}')

                if self._compute_inner_users and name == 'Test':
                    roc_auc_inner = roc_auc_score(
                        y.numpy()[self.test_inner_users, :],
                        predictions[self.test_inner_users, :],
                        average=avg
                    )
                    if self.use_mlflow:
                        mlflow.log_metric(f"test_roc_auc_{avg}_inner", roc_auc_inner, step=step)
                    else:
                        logger.info(f'Test roc_auc_{avg}_inner on step {step}: {roc_auc_inner}')

            roc_auc_list = []
            roc_auc_list_inner = []
            for i in range(self.n_labels):
                roc_auc_list.append(roc_auc_score(y.numpy()[:, i], predictions[:, i]))
                if self._compute_inner_users and name == 'Test':
                    roc_auc_list_inner.append(
                        roc_auc_score(y.numpy()[self.test_inner_users, i], predictions[self.test_inner_users, i])
                    )

            for i, cls in enumerate(self.classes_idx):
                if self.use_mlflow:
                    mlflow.log_metric(f"{name.lower()}_roc_auc_cls_{cls}", roc_auc_list[i])
                    if self._compute_inner_users and name == 'Test':
                        mlflow.log_metric(f"test_roc_auc_cls_{cls}_inner", roc_auc_list_inner[i])
                else:
                    logger.info(f"{name.lower()}_roc_auc_cls_{cls}: {roc_auc_list[i]}")
                    if self._compute_inner_users and name == 'Test':
                        logger.info(f"test_roc_auc_cls_{cls}_inner: {roc_auc_list_inner[i]}")

        else:
            roc_auc = roc_auc_score(y.numpy(), predictions)
            if self.use_mlflow:
                mlflow.log_metric(f"{name.lower()}_roc_auc", roc_auc, step=step)
            else:
                logger.info(f"{name} ROC AUC on step {step}: {roc_auc}")

            p_train, r_train, _ = precision_recall_curve(y.numpy(), predictions)
            pr_auc_train = auc(r_train, p_train)
            if self.use_mlflow:
                mlflow.log_metric(f"{name.lower()}_pr_auc", pr_auc_train, step=step)
            else:
                logger.info(f'{name} Precision-Recall score on step: {step}: {pr_auc_train}')


class PartySingleLinreg(PartySingle):
    model_name = 'linreg'

    def update_weights(
            self,
            x: DataTensor,
            y: DataTensor,
    ) -> None:
        self._model.update_weights(x, rhs=y)

    def initialize_model(self):
        self._model = LinearRegressionBatch(
            input_dim=self._dataset[self._data_params.train_split][self._data_params.features_key].shape[1],
            output_dim=1, reg_lambda=0.5
        )

    def compute_predictions(
            self,
            is_test: bool = False,
    ) -> DataTensor:
        features = self.x_test if is_test else self.x_train
        return self._model.predict(features).detach().numpy()


# class PartySingleLogreg(PartySingle):
#     model_name = 'logreg'
#
#     def update_weights(
#             self,
#             x: DataTensor,
#             y: DataTensor,
#     ):
#         self._model.update_weights(x, y, is_single=True)
#
#     def _init_model(self):
#         self._model = LogisticRegressionBatch(
#             input_dim=self.input_dims[self.member_id],
#             output_dim=self.n_labels,
#             learning_rate=self.learning_rate,
#             class_weights=None,  # TODO add!
#             init_weights=0.005
#         )
#
#     def compute_predictions(
#             self,
#             is_test: bool = False,
#     ) -> DataTensor:
#         features = self._x_test if is_test else self._x_train
#         return torch.sigmoid(self._model.predict(features)).detach().numpy()


