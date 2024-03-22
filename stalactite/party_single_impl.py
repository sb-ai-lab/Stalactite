"""Centralized (non-VFL) experiment runner.

This module contains the implementation of a centralized experiment for federated learning.
It includes classes for logistic regression and linear regression centralized models training.
"""

from abc import abstractmethod
from typing import List, Optional
import logging

import mlflow
import torch

from torch.optim import SGD
from torch import nn
from sklearn.metrics import mean_absolute_error, roc_auc_score, root_mean_squared_error

from stalactite.base import DataTensor
from stalactite.batching import Batcher, ListBatcher
from stalactite.metrics import ComputeAccuracy_numpy
from stalactite.models import LogisticRegressionBatch, LinearRegressionBatch, EfficientNet, MLP, ResNet
from stalactite.models.split_learning import EfficientNetTop, EfficientNetBottom, MLPTop, MLPBottom, ResNetBottom, ResNetTop
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
            target_uids: list = None,
            model_params: dict = None

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
        self.processor = processor
        self._model = None
        self.is_initialized = False
        self.is_finalized = False
        self.target_uids = target_uids
        self._uid2tensor_idx = {uid: i for i, uid in enumerate(self.target_uids)}
        self._model_params = model_params
        self._optimizer = None
        self._criterion = None
        self._activation = None
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
            tensor_idx = [self._uid2tensor_idx[uid] for uid in batch]

            x = self.x_train[tensor_idx]
            y = self.target[tensor_idx]

            self.update_weights(x=x, y=y, optimizer=self._optimizer, criterion=self._criterion)

            predictions = self.compute_predictions(is_test=False, use_activation=True).detach().numpy()
            predictions_test = self.compute_predictions(is_test=True, use_activation=True).detach().numpy()

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
        return sorted(x for x in self.target_uids)

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

    def update_weights(
            self,
            x: DataTensor,
            y: DataTensor,
            optimizer: torch.optim.Optimizer = None,
            criterion: Optional[torch.nn.Module] = None
    ) -> None:
        self._model.update_weights(x, y, is_single=True, optimizer=self._optimizer, criterion=self._criterion)

    def compute_predictions(
            self,
            is_test: bool = False,
            use_activation: bool = False
    ) -> DataTensor:
        features = self.x_test if is_test else self.x_train
        predictions = self._model.predict(features)
        if use_activation:
            predictions = self._activation(predictions)
        return predictions

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
        a = torch.isnan(self.x_train).any()  # todo: remove
        self.x_test = self._dataset[self.processor.data_params.test_split][self.processor.data_params.features_key]
        b = torch.isnan(self.x_test).any()  # todo: remove
        self.target = self._dataset[self.processor.data_params.train_split][self.processor.data_params.label_key]
        self.test_target = self._dataset[self.processor.data_params.test_split][self.processor.data_params.label_key]

        self.class_weights = self.processor.get_class_weights() \
            if self.processor.common_params.use_class_weights else None

        self._data_params = self.processor.data_params
        self._common_params = self.processor.common_params
        if torch.equal(torch.unique(self.target), torch.tensor([0, 1])) or torch.max(self.target).item() <= 1:
            self._activation = nn.Sigmoid()
            self.binary = True
        else:
            self._activation = nn.Softmax(dim=1)
            self.binary = False

        self.initialize_model()
        self.is_initialized = True
        logger.info("Centralized experiment is initialized")

    def finalize(self) -> None:
        """Finalize the experiment.

        :return: None
        """
        self.is_finalized = True
        logger.info("Experiment has finished")

    def report_metrics(self, y: DataTensor, predictions: DataTensor, name: str, step: int) -> None:
        """Report metrics based on target values, predictions, and name.

        Compute main metrics, if `use_mlflow` parameter was set to true, log them to MlFLow,
        otherwise, log them to stdout.

        :param y: Target values.
        :param predictions: Predicted values.
        :param name: Name for the metrics report (`Train`, `Test`).
        :param step: Current step or iteration.
        :return: None
        """

        if self.binary:
            for avg in ["macro", "micro"]:
                try:
                    roc_auc = roc_auc_score(y, predictions, average=avg)
                except ValueError:
                    roc_auc = 0

                rmse = root_mean_squared_error(y, predictions)

                logger.info(f'{name} RMSE on step {step}: {rmse}')
                logger.info(f'{name} ROC AUC {avg} on step {step}: {roc_auc}')
                if self.use_mlflow:
                    mlflow.log_metric(f"{name.lower()}_roc_auc_{avg}", roc_auc, step=step)
                    mlflow.log_metric(f"{name.lower()}_rmse", rmse, step=step)

        else:
            avg = "macro"
            try:
                roc_auc = roc_auc_score(y, predictions, average=avg, multi_class="ovr")
            except ValueError:
                roc_auc = 0

            if self.use_mlflow:
                mlflow.log_metric(f"{name.lower()}_roc_auc_{avg}", roc_auc, step=step)


class PartySingleLinreg(PartySingle):
    """ Single-agent (centralized) experiment runner class, implementing the regression algorithm using linear
    regression model. """
    model_name = 'linreg'

    def update_weights(
            self,
            x: DataTensor,
            y: DataTensor,
            optimizer: torch.optim.Optimizer = None,
            criterion: Optional[torch.nn.Module] = None
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
            use_activation: bool = False
    ) -> DataTensor:
        features = self.x_test if is_test else self.x_train
        return self._model.predict(features)


class PartySingleLogreg(PartySingle):
    """ Single-agent (centralized) experiment runner class, implementing the classification algorithm using logistic
    regression model. """
    model_name = 'logreg'

    def initialize_model(self):
        self._model = LogisticRegressionBatch(
            input_dim=self._dataset[self._data_params.train_split][self._data_params.features_key].shape[1],
            **self._model_params
        )

        self._optimizer = torch.optim.SGD(
            self._model.parameters(),
            lr=self._common_params.learning_rate,
            momentum=self._common_params.momentum,
            weight_decay=0.02 #self._common_params.weights_decay
        )

        self._criterion = torch.nn.BCEWithLogitsLoss(pos_weight=self.class_weights)
        self._activation = nn.Sigmoid()


class PartySingleLogregMulticlass(PartySingleLogreg):

    def initialize_model(self):
        self._model = LogisticRegressionBatch(
            input_dim=self._dataset[self._data_params.train_split][self._data_params.features_key].shape[1],
            **self._model_params
        )

        self._optimizer = torch.optim.SGD(
            self._model.parameters(),
            lr=self._common_params.learning_rate,
            momentum=self._common_params.momentum
        )

        self._criterion = torch.nn.CrossEntropyLoss(weight=self.class_weights)
        self._activation = nn.Softmax(dim=1)


class PartySingleEfficientNet(PartySingleLogregMulticlass):
    def initialize_model(self):
        self._model = EfficientNet(
            width_mult=0.1,
            depth_mult=0.1,
            dropout=0.2,
            num_classes=10,
            init_weights=None)
        self._optimizer = torch.optim.SGD(
            self._model.parameters(),
            lr=self._common_params.learning_rate,
            momentum=self._common_params.momentum
        )

        if self.use_mlflow:
            mlflow.log_param("model_type", "base")

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


class PartySingleEfficientNetSplitNN(PartySingleLogregMulticlass):
    def initialize_model(self):
        self._model_top = EfficientNetTop(
            input_dim=128,
            dropout=0.2,
            num_classes=10,
            init_weights=None)

        self._model_bottom = EfficientNetBottom(
             width_mult=0.1,
             depth_mult=0.1,
             init_weights=None)


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


class PartySingleMLP(PartySingle):

    model_name = 'mlp'

    def update_weights(
            self,
            x: DataTensor,
            y: DataTensor,
            optimizer: torch.optim.Optimizer = None,
            criterion: Optional[torch.nn.Module] = None
    ) -> None:
        self._model.update_weights(x, y, is_single=True, optimizer=self._optimizer, criterion=self._criterion)

    def initialize_model(self):

        self._model = MLP(
            input_dim=self._dataset[self._data_params.train_split][self._data_params.features_key].shape[1],
            **self._model_params
        )

        self._optimizer = torch.optim.SGD(
            self._model.parameters(),
            lr=self._common_params.learning_rate,
            momentum=self._common_params.momentum
        )

        self._criterion = torch.nn.BCEWithLogitsLoss(pos_weight=self.class_weights) if self.binary else torch.nn.CrossEntropyLoss(weight=self.class_weights)


class PartySingleMLPSplitNN(PartySingleLogregMulticlass):
    def initialize_model(self):
        init_weights = None
        self._model_top = MLPTop(
            input_dim=100,
            output_dim=1,
            init_weights=init_weights,
            class_weights=self.class_weights)

        self._model_bottom = MLPBottom(
            input_dim=1356,
            hidden_channels=[1000, 300, 100],
            init_weights=init_weights)


        self._optimizer = torch.optim.SGD([
            {"params": self._model_top.parameters()},
            {"params": self._model_bottom.parameters()}
        ],
            lr=self._common_params.learning_rate,
            momentum=self._common_params.momentum
        )

        if self.use_mlflow:
            mlflow.log_param("model_type", "divided")
            mlflow.log_param("init_weights", init_weights)

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

        return torch.sigmoid(predictions).detach().numpy()


class PartySingleResNet(PartySingle):

    model_name = 'resnet'

    def initialize_model(self):
        self._model = ResNet(
            input_dim=self._dataset[self._data_params.train_split][self._data_params.features_key].shape[1],
            **self._model_params,
            )

        self._optimizer = torch.optim.SGD(
            self._model.parameters(),
            lr=self._common_params.learning_rate,
            momentum=self._common_params.momentum
        )

        self._criterion = torch.nn.BCEWithLogitsLoss(
            pos_weight=self.class_weights) if self.binary else torch.nn.CrossEntropyLoss(weight=self.class_weights)

    # def compute_predictions(
    #         self,
    #         is_test: bool = False,
    # ) -> DataTensor:
    #     features = self.x_test if is_test else self.x_train
    #     return torch.sigmoid(self._model.predict(features)).detach().numpy()


class PartySingleResNetSplitNN(PartySingleLogregMulticlass):
    def initialize_model(self):
        init_weights = None
        self._model_top = ResNetTop(
            input_dim=1356,
            output_dim=1,
            init_weights=init_weights,
            use_bn=True,
        )

        self._model_bottom = ResNetBottom(
            input_dim=1356,
            hid_factor=[1, 1],
            init_weights=init_weights)

        self._optimizer = torch.optim.SGD([
            {"params": self._model_top.parameters()},
            {"params": self._model_bottom.parameters()}
        ],
            lr=self._common_params.learning_rate,
            momentum=self._common_params.momentum
        )

        self._criterion = torch.nn.BCEWithLogitsLoss(pos_weight=self.class_weights)

        if self.use_mlflow:
            mlflow.log_param("model_type", "divided")
            mlflow.log_param("init_weights", init_weights)

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
            optimizer=self._optimizer,
            criterion=self._criterion
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

        return torch.sigmoid(predictions).detach().numpy()
