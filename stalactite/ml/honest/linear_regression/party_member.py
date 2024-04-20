import logging
import os
from abc import ABC
from typing import Optional, Any
import math

import torch
from torch import nn
import numpy as np

from stalactite.base import RecordsBatch, DataTensor
from stalactite.ml.honest.base import HonestPartyMember
from stalactite.models import LinearRegressionBatch
from stalactite.utils import init_linear_np

logger = logging.getLogger(__name__)

class HonestPartyMemberLinReg(HonestPartyMember):

    def initialize_model_from_params(self, **model_params) -> Any:
        return LinearRegressionBatch(**model_params)

    def initialize_model(self, do_load_model: bool = False) -> None:
        """ Initialize the model based on the specified model name. """
        if do_load_model:
            self._model = self.load_model()

        else:
            self._model = LinearRegressionBatch(
                input_dim=self._dataset[self._data_params.train_split][self._data_params.features_key].shape[1],
                **self._model_params
            )

            init_linear_np(self._model.linear, seed=self.seed)


    def initialize_optimizer(self) -> None:
        pass

    def update_weights(self, uids: RecordsBatch, upd: DataTensor) -> None:
        """ Update model weights based on input features and target values.

        :param uids: Batch of record unique identifiers.
        :param upd: Updated model weights.
        """
        logger.info("Member %s: updating weights. Incoming tensor: %s" % (self.id, tuple(upd.size())))
        self.check_if_ready()
        tensor_idx = [self.uid2tensor_idx[uid] for uid in uids]
        X_train = self._dataset[self._data_params.train_split][self._data_params.features_key][tensor_idx, :]
        self._model.update_weights(X_train, upd, optimizer=self._optimizer)
        logger.info("Member %s: successfully updated weights" % self.id)

    def predict(self, uids: Optional[RecordsBatch], is_infer: bool = False) -> DataTensor:
        """ Make predictions using the current model.

        :param uids: Batch of record unique identifiers.
        :param is_infer: Flag indicating whether to use the test data.

        :return: Model predictions.
        """
        logger.info("Member %s: predicting." % (self.id))
        self.check_if_ready()
        _uid2tensor_idx = self.uid2tensor_idx_test if is_infer else self.uid2tensor_idx
        tensor_idx = [_uid2tensor_idx[uid] for uid in uids] if uids else None
        if is_infer:
            logger.info("Member %s: using test data" % self.id)
            if uids is None:
                X = self._dataset[self._data_params.test_split][self._data_params.features_key]
            else:
                X = self._dataset[self._data_params.test_split][self._data_params.features_key][tensor_idx, :]
        else:
            X = self._dataset[self._data_params.train_split][self._data_params.features_key][tensor_idx, :]
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
        self.check_if_ready()
        if previous_batch is not None:
            self.update_weights(uids=previous_batch, upd=upd)
        predictions = self.predict(batch)
        self.iterations_counter += 1
        logger.info("Member %s: updated and predicted." % self.id)
        return predictions
