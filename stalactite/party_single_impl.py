"""Centralized (non-VFL) experiment runner.

This module contains the implementation of a centralized experiment for federated learning.
It includes classes for logistic regression and linear regression centralized models training.
"""

from abc import abstractmethod
from typing import List, Optional
import logging

import mlflow
import torch

from sklearn.metrics import mean_absolute_error, roc_auc_score

from stalactite.base import DataTensor
from stalactite.batching import Batcher, ListBatcher
from stalactite.metrics import ComputeAccuracy_numpy
from stalactite.models import LogisticRegressionBatch, LinearRegressionBatch
from stalactite.data_preprocessors.base_preprocessor import DataPreprocessor

logger = logging.getLogger(__name__)


class PartySingle:
    """ Single-agent (centralized) experiment runner class. """

    model_name: str
    member_id = 0

    def __init__(
            self,
            epochs: int,
            batch_size: int,
            processor: Optional[DataPreprocessor] = None,
            use_mlflow: bool = False,
            report_train_metrics_iteration: int = 1,
            report_test_metrics_iteration: int = 1,
            learning_rate: float = 0.01,

    ) -> None:
        """Initialize PartySingle instance.

        :param epochs: Number of epochs to train a model.
        :param batch_size: Batch size used for training.
        :param processor: Data preprocessor.
        :param use_mlflow: Flag indicating whether to log metrics to MlFlow.
        :param report_train_metrics_iteration: Number of iteration steps between reporting metrics on train dataset
               split.
        :param report_test_metrics_iteration: Number of iteration steps between reporting metrics on test dataset split.
        :param learning_rate: Learning rate.
        :return: None
        """
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

    def run(self) -> None:
        """ Run centralized experiment.

        :return: None
        """
        self.initialize()
        uids = self.synchronize_uids()
        self.loop(batcher=self.make_batcher(uids=uids))
        self.finalize()

    def loop(self, batcher: Batcher) -> None:
        """ Perform training iterations using the given make_batcher.

        :param batcher: An iterable batch generator used for training.
        :return: None
        """
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
        """ Get the uuids for current experiment.

        :return: List of UUIDs as strings.
        """
        return [str(x) for x in range(self.target.shape[0])]

    def make_batcher(self, uids: List[str]) -> Batcher:
        """ Create a make_batcher based on the provided UUIDs.

        :param uids: List of UUIDs.
        :return: A Batcher object.
        """
        return ListBatcher(
            epochs=self.epochs,
            members=[str(x) for x in range(10)],
            uids=uids,
            batch_size=self.batch_size,
            shuffle=False
        )

    @abstractmethod
    def update_weights(self, x: DataTensor, y: DataTensor):
        """Update the model weights based on the input features and target values.

        :param x: Input features.
        :param y: Target values.
        :return: None
        """
        ...

    @abstractmethod
    def compute_predictions(self, is_test: bool = False) -> DataTensor:
        """Compute predictions using the current model.

        :param is_test: Flag indicating whether to compute predictions for the test set.
        :return: Predicted values.
        """
        ...

    @abstractmethod
    def initialize_model(self) -> None:
        """Initialize the model to train.

        :return: None
        """
        ...

    def initialize(self) -> None:
        """ Initialize the centralized experiment.

        :return: None
        """
        logger.info("Centralized experiment initializing")

        self._dataset = self.processor.fit_transform()

        self.x_train = self._dataset[self.processor.data_params.train_split][self.processor.data_params.features_key]
        self.x_test = self._dataset[self.processor.data_params.test_split][self.processor.data_params.features_key]
        self.target = self._dataset[self.processor.data_params.train_split][self.processor.data_params.label_key]
        self.test_target = self._dataset[self.processor.data_params.test_split][self.processor.data_params.label_key]

        self.class_weights = self.processor.get_class_weights() \
            if self.processor.common_params.use_class_weights else None

        self._data_params = self.processor.data_params
        self._common_params = self.processor.common_params
        self.n_labels = 19
        self.initialize_model()
        self.is_initialized = True
        logger.info("Centralized experiment is initialized")

    def finalize(self) -> None:
        """Finalize the experiment.

        :return: None
        """
        self.is_finalized = True
        logger.info("Experiment has finished")

    def report_metrics(self, y: DataTensor, predictions: DataTensor, name: str, step: int):
        """Report metrics based on target values, predictions, and name.

        Compute main metrics, if `use_mlflow` parameter was set to true, log them to MlFLow,
        otherwise, log them to stdout.

        :param y: Target values.
        :param predictions: Predicted values.
        :param name: Name for the metrics report (`Train`, `Test`).
        :param step: Current step or iteration.
        :return: None
        """
        acc = ComputeAccuracy_numpy(is_linreg=self.model_name == "linreg", is_multilabel=True)

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
                try:
                    roc_auc = roc_auc_score(y.numpy(), predictions, average=avg)
                except ValueError:
                    roc_auc = 0
                if self.use_mlflow:
                    mlflow.log_metric(f"{name.lower()}_roc_auc_{avg}", roc_auc, step=step)
                else:
                    logger.info(f'{name} roc_auc_{avg} on step {step}: {roc_auc}')

        else:
            try:
                roc_auc = roc_auc_score(y.numpy(), predictions)
            except ValueError:
                roc_auc = 0
            if self.use_mlflow:
                mlflow.log_metric(f"{name.lower()}_roc_auc", roc_auc, step=step)
            else:
                logger.info(f"{name} ROC AUC on step {step}: {roc_auc}")


class PartySingleLinreg(PartySingle):
    """ Single-agent (centralized) experiment runner class, implementing the regression algorithm using linear
    regression model. """
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


class PartySingleLogreg(PartySingle):
    """ Single-agent (centralized) experiment runner class, implementing the classification algorithm using logistic
    regression model. """
    model_name = 'logreg'

    def update_weights(
            self,
            x: DataTensor,
            y: DataTensor,
    ):
        self._model.update_weights(x, y, is_single=True)

    def initialize_model(self):
        self._model = LogisticRegressionBatch(
            input_dim=self._dataset[self._data_params.train_split][self._data_params.features_key].shape[1],
            output_dim=self._dataset[self._data_params.train_split][self._data_params.label_key].shape[1],
            learning_rate=self._common_params.learning_rate,
            class_weights=self.class_weights,
            init_weights=0.005)

    def compute_predictions(
            self,
            is_test: bool = False,
    ) -> DataTensor:
        features = self.x_test if is_test else self.x_train
        return torch.sigmoid(self._model.predict(features)).detach().numpy()
