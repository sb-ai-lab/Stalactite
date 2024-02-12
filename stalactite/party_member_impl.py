import logging
from typing import List, Optional, Tuple

import torch
import scipy as sp

from stalactite.base import Batcher, DataTensor, PartyMember, RecordsBatch
from stalactite.batching import ListBatcher, ConsecutiveListBatcher
from stalactite.models import LinearRegressionBatch, LogisticRegressionBatch
from stalactite.models.split_learning.efficientnet_bottom import EfficientNetBottom

logger = logging.getLogger(__name__)


class PartyMemberImpl(PartyMember):
    """ Implementation class of the PartyMember used for local and distributed VFL training. """

    def __init__(
        self,
        uid: str,
        epochs: int,
        batch_size: int,
        member_record_uids: List[str],
        model_name: str,
        report_train_metrics_iteration: int,
        report_test_metrics_iteration: int,
        processor=None,
        is_consequently: bool = False,
        members: Optional[list[str]] = None,
    ) -> None:
        """
        Initialize PartyMemberImpl.

        :param uid: Unique identifier for the party member.
        :param epochs: Number of training epochs.
        :param batch_size: Size of the training batch.
        :param member_record_uids: List of unique identifiers of the dataset rows to use.
        :param model_name: Name of the model to be used.
        :param report_train_metrics_iteration: Number of iterations between reporting metrics on the train dataset.
        :param report_test_metrics_iteration: Number of iterations between reporting metrics on the test dataset.
        :param processor: Optional data processor.
        :param is_consequently: Flag indicating whether to use the consequent implementation (including the batcher).
        :param members: List of the members if the algorithm is consequent.
        """
        self.id = uid
        self.epochs = epochs
        self._batch_size = batch_size
        self._uids = member_record_uids
        self._uids_to_use: Optional[List[str]] = None
        self.is_initialized = False
        self.is_finalized = False
        self.iterations_counter = 0
        self._model_name = model_name
        self.report_train_metrics_iteration = report_train_metrics_iteration
        self.report_test_metrics_iteration = report_test_metrics_iteration
        self.processor = processor
        self._batcher = None
        self.is_consequently = is_consequently
        self.members = members

        if self.is_consequently:
            if self.members is None:
                raise ValueError('If consequent algorithm is initialized, the members must be passed.')

    def _create_batcher(self, epochs: int, uids: List[str], batch_size: int) -> None:
        """Create a batcher for training.

        :param epochs: Number of training epochs.
        :param uids: List of unique identifiers for dataset rows.
        :param batch_size: Size of the training batch.
        """
        logger.info("Member %s: making a batcher for uids" % (self.id))
        self._check_if_ready()
        if not self.is_consequently:
            self._batcher = ListBatcher(epochs=epochs, members=None, uids=uids, batch_size=batch_size)
        else:
            self._batcher = ConsecutiveListBatcher(
                epochs=self.epochs, members=self.members, uids=uids, batch_size=self._batch_size
            )

    @property
    def batcher(self) -> Batcher:
        """ Get the batcher for training.
        Initialize and return the batcher if it has not been initialized yet, otherwise, return created batcher.

        :return: Batcher instance.
        """
        if self._batcher is None:
            if self._uids_to_use is None:
                raise RuntimeError("Cannot create batcher, you must `register_records_uids` first.")
            self._create_batcher(epochs=self.epochs, uids=self._uids_to_use, batch_size=self._batch_size)
        else:
            logger.info("Member %s: using created batcher" % (self.id))
        return self._batcher

    def records_uids(self) -> List[str]:
        """ Get the list of existing dataset unique identifiers.

        :return: List of unique identifiers.
        """
        logger.info("Member %s: reporting existing record uids" % self.id)
        return self._uids

    def register_records_uids(self, uids: List[str]) -> None:
        """ Register unique identifiers to be used.

        :param uids: List of unique identifiers.
        :return: None
        """
        logger.info("Member %s: registering %s uids to be used." % (self.id, len(uids)))
        self._uids_to_use = uids

    def initialize_model(self) -> None:
        """ Initialize the model based on the specified model name. """
        if self._model_name == "linreg":
            self._model = LinearRegressionBatch(
                input_dim=self._dataset[self._data_params.train_split][self._data_params.features_key].shape[1],
                output_dim=1, reg_lambda=0.5
            )
        elif self._model_name == "logreg":
            self._model = LogisticRegressionBatch(
                input_dim=self._dataset[self._data_params.train_split][self._data_params.features_key].shape[1],
                output_dim=self._dataset[self._data_params.train_split][self._data_params.label_key].shape[1],
                learning_rate=self._common_params.learning_rate,
                class_weights=None,
                init_weights=0.005)
        elif self._model_name == "efficientnet":
            self._model = EfficientNetBottom(
                width_mult=1.0,
                depth_mult=1.0,
                stochastic_depth_prob=0.2)
        else:
            raise ValueError("unknown model %s" % self._model_name)

    def initialize_optimizer(self) -> None:
        self._optimizer = torch.optim.SGD([
            {"params": self._model.parameters()},
        ],
            lr=self._common_params.learning_rate,
            momentum=self._common_params.momentum
        )

    def initialize(self) -> None:
        """ Initialize the party member. """

        logger.info("Member %s: initializing" % self.id)
        self._dataset = self.processor.fit_transform()
        self._data_params = self.processor.data_params
        self._common_params = self.processor.common_params
        self.initialize_model()
        self.initialize_optimizer()
        self.is_initialized = True
        logger.info("Member %s: has been initialized" % self.id)

    def finalize(self) -> None:
        """ Finalize the party member. """
        logger.info("Member %s: finalizing" % self.id)
        self._check_if_ready()
        self.is_finalized = True
        logger.info("Member %s: has been finalized" % self.id)

    def _prepare_data(self, uids: RecordsBatch) -> Tuple:
        """ Prepare data for training.

        :param uids: Batch of record unique identifiers.
        :return: Tuple of three SVD matrices.
        """
        X_train = self._dataset[self._data_params.train_split][self._data_params.features_key][[int(x) for x in uids]]
        U, S, Vh = sp.linalg.svd(X_train.numpy(), full_matrices=False, overwrite_a=False, check_finite=False)
        return U, S, Vh

    def update_weights(self, uids: RecordsBatch, upd: DataTensor) -> None:
        """ Update model weights based on input features and target values.

        :param uids: Batch of record unique identifiers.
        :param upd: Updated model weights.
        """
        logger.info("Member %s: updating weights. Incoming tensor: %s" % (self.id, tuple(upd.size())))
        self._check_if_ready()
        X_train = self._dataset[self._data_params.train_split][self._data_params.features_key][[int(x) for x in uids]]
        self._model.update_weights(X_train, upd, optimizer=self._optimizer)
        logger.info("Member %s: successfully updated weights" % self.id)

    def predict(self, uids: RecordsBatch, use_test: bool = False) -> DataTensor:
        """ Make predictions using the current model.

        :param uids: Batch of record unique identifiers.
        :param use_test: Flag indicating whether to use the test data.

        :return: Model predictions.
        """
        logger.info("Member %s: predicting. Batch size: %s" % (self.id, len(uids)))
        self._check_if_ready()
        if use_test:
            logger.info("Member %s: using test data" % self.id)
            X = self._dataset[self._data_params.test_split][self._data_params.features_key]
        else:
            X = self._dataset[self._data_params.train_split][self._data_params.features_key][[int(x) for x in uids]]
        predictions = self._model.predict(X)
        logger.info("Member %s: made predictions." % self.id)
        return predictions

    def update_predict(self, upd: DataTensor, previous_batch: RecordsBatch, batch: RecordsBatch) -> DataTensor:
        """ Update model weights and make predictions.

        :param upd: Updated model weights.
        :param previous_batch: Previous batch of record unique identifiers.
        :param batch: Current batch of record unique identifiers.

        :return: Model predictions.
        """
        logger.info("Member %s: updating and predicting." % self.id)
        self._check_if_ready()
        uids = previous_batch if previous_batch is not None else batch
        self.update_weights(uids=uids, upd=upd)
        predictions = self.predict(batch)
        self.iterations_counter += 1
        logger.info("Member %s: updated and predicted." % self.id)
        return predictions

    def _check_if_ready(self):
        """ Check if the party member is ready for operations.

        Raise a RuntimeError if experiment has not been initialized or has already finished.
        """
        if not self.is_initialized and not self.is_finalized:
            raise RuntimeError("The member has not been initialized")
