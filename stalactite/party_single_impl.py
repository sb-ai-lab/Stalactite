"""Centralized (non-VFL) experiment runner.

This module contains the implementation of a centralized experiment for federated learning.
It includes classes for logistic regression and linear regression centralized models training.
"""

from abc import abstractmethod
from typing import List, Optional
import logging

import mlflow
import torch

from torchsummary import summary
from sklearn.metrics import mean_absolute_error, roc_auc_score

from stalactite.base import DataTensor
from stalactite.batching import Batcher, ListBatcher
from stalactite.metrics import ComputeAccuracy_numpy
from stalactite.models import LogisticRegressionBatch, LinearRegressionBatch, EfficientNet
from stalactite.models.split_learning import EfficientNetTop, EfficientNetBottom
from stalactite.data_preprocessors.base_preprocessor import DataPreprocessor

logger = logging.getLogger(__name__)

logger.setLevel(logging.DEBUG)
sh = logging.StreamHandler()
sh.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
logger.addHandler(sh)


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
            target_uids: list = None

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
        self.target_uids = target_uids

    def run(self) -> None:
        """ Run centralized experiment.

        :return: None
        """
        self.initialize()
        uids = self.synchronize_uids()
        self.loop(batcher=self.make_batcher(uids=uids))
        self.finalize()

    def loop(self, batcher: Batcher) -> None:
        """ Perform training iterations using the given batcher.

        :param batcher: An iterable batch generator used for training.
        :return: None
        """

        for titer in batcher:
            step = titer.seq_num
            logger.debug(f"batch: {step}")
            batch = titer.batch
            tensor_idx = [int(x) for x in batch]

            x = self.x_train[tensor_idx]
            y = self.target[tensor_idx][:, 6] #TODO: REMOVE IT

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
        return [str(x) for x in self.target_uids]

    def make_batcher(self, uids: List[str]) -> Batcher:
        """ Create a batcher based on the provided UUIDs.

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
        self.n_labels = 19  # todo: refactor it
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

        y = y[:, 6]  # todo: remove
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
            output_dim=1 , #self._dataset[self._data_params.train_split][self._data_params.label_key].shape[1],
            learning_rate=self._common_params.learning_rate,
            class_weights=None,#self.class_weights[6],
            init_weights=0.005)

    def compute_predictions(
            self,
            is_test: bool = False,
    ) -> DataTensor:
        features = self.x_test if is_test else self.x_train
        return torch.sigmoid(self._model.predict(features)).detach().numpy()


class PartySingleLogregMulticlass(PartySingleLogreg):
    def initialize_model(self):
        self._model = LogisticRegressionBatch(
            input_dim=self._dataset[self._data_params.train_split][self._data_params.features_key].shape[1],
            output_dim=10,  # todo: make it more beautiful
            learning_rate=self._common_params.learning_rate,
            class_weights=None,
            init_weights=0.005,
            multilabel=False)

    def compute_predictions(
            self,
            is_test: bool = False,
    ) -> DataTensor:
        features = self.x_test if is_test else self.x_train
        return torch.softmax(self._model.predict(features), dim=1).detach().numpy()

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
        for avg in ["macro"]: #, "micro"]:
            # try:
            roc_auc = roc_auc_score(y.numpy(), predictions, average=avg, multi_class="ovr")
            # except ValueError:
            #     roc_auc = 0
            if self.use_mlflow:
                mlflow.log_metric(f"{name.lower()}_roc_auc_{avg}", roc_auc, step=step)
            else:
                logger.info(f'{name} roc_auc_{avg} on step {step}: {roc_auc}')


class PartySingleEfficientNet(PartySingleLogregMulticlass):
    def initialize_model(self):
        self._model = EfficientNet(
            width_mult=0.1,
            depth_mult=0.1,
            dropout=0.2,
            num_classes=10,
            init_weights=None)
        logger.info(summary(self._model, (1, 28, 28), device="cpu"))
        self._optimizer = torch.optim.SGD(
            self._model.parameters(),
            lr=self._common_params.learning_rate,
            momentum=self._common_params.momentum
        )

        if self.use_mlflow:
            mlflow.log_param("model_type", "base")
    def update_weights(
            self,
            x: DataTensor,
            y: DataTensor,
    ):
        self._model.update_weights(x, y, is_single=True, optimizer=self._optimizer)

    def loop(self, batcher: Batcher) -> None:
        """ Perform training iterations using the given batcher.

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

    def compute_predictions(
            self,
            is_test: bool = False,
    ) -> DataTensor:
        features = self.x_test if is_test else self.x_train
        return torch.softmax(self._model.predict(features), dim=1).detach().numpy()


class PartySingleEfficientNetSplitNN(PartySingleLogregMulticlass):
    def initialize_model(self):
        self._model_top = EfficientNetTop(
            input_dim=128,  # todo: determine in somehow
            dropout=0.2,
            num_classes=10,
            init_weights=None)
        logger.info(summary(self._model_top, (128, 1, 1) , device="cpu"))

        self._model_bottom = EfficientNetBottom(
             width_mult=0.1,
             depth_mult=0.1,
             init_weights=None)

        logger.info(summary(self._model_bottom, (1, 28, 28), device="cpu"))

        self._optimizer = torch.optim.SGD([
            {"params": self._model_top.parameters()},
            {"params": self._model_bottom.parameters()}
        ],
            lr=self._common_params.learning_rate,
            momentum=self._common_params.momentum
        )

        if self.use_mlflow:
            mlflow.log_param("model_type", "divided")

    def update_weights(
            self,
            x: DataTensor,
            y: DataTensor,
    ):
        bottom_model_outputs = self._model_bottom.forward(x)
        top_grads = self._model_top.update_weights(
            x=bottom_model_outputs,
            gradients=y,
            is_single=True,
            optimizer=self._optimizer
        )
        self._model_bottom.update_weights(x=x, gradients=top_grads, optimizer=self._optimizer)

    def loop(self, batcher: Batcher) -> None:
        """ Perform training iterations using the given batcher.

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

    def compute_predictions(
            self,
            is_test: bool = False,
    ) -> DataTensor:

        features = self.x_test if is_test else self.x_train
        bottom_model_outputs = self._model_bottom.forward(features)
        predictions = self._model_top.predict(bottom_model_outputs)

        return torch.softmax(predictions, dim=1).detach().numpy()
