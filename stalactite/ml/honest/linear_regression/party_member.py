import logging
from typing import Optional, Any

from stalactite.base import RecordsBatch, DataTensor
from stalactite.ml.honest.base import HonestPartyMember
from stalactite.models import LinearRegressionBatch
from stalactite.utils import init_linear_np

logger = logging.getLogger(__name__)


class HonestPartyMemberLinReg(HonestPartyMember):

    def initialize_model_from_params(self, **model_params) -> Any:
        return LinearRegressionBatch(**model_params).to(self.device)

    def initialize_model(self, do_load_model: bool = False) -> None:
        """ Initialize the model based on the specified model name. """
        logger.info(f"Member {self.id} initializes model on device: {self.device}")
        logger.debug(f"Model is loaded from path: {do_load_model}")
        if do_load_model:
            self._model = self.load_model().to(self.device)
        else:
            self._model = LinearRegressionBatch(
                input_dim=self._dataset[self._data_params.train_split][self._data_params.features_key].shape[1],
                **self._model_params
            ).to(self.device)

            init_linear_np(self._model.linear, seed=self.seed)
            self._model.linear.to(self.device)

    def initialize_optimizer(self) -> None:
        pass

    def update_weights(self, uids: RecordsBatch, upd: DataTensor) -> None:
        """ Update model weights based on input features and target values.

        :param uids: Batch of record unique identifiers.
        :param upd: Updated model weights.
        """
        logger.info(f"Member {self.id}: updating weights. Incoming tensor: {upd.size()}")
        self.check_if_ready()
        tensor_idx = [self.uid2tensor_idx[uid] for uid in uids]
        X_train = self.device_dataset_train_split[tensor_idx, :]
        self._model.update_weights(X_train, upd.to(self.device), optimizer=self._optimizer)
        self.move_model_to_device()
        logger.debug(f"Member {self.id}: successfully updated weights")

    def move_model_to_device(self):
        self._model.linear.weight.to(self.device)

    def predict(self, uids: Optional[RecordsBatch], is_infer: bool = False) -> DataTensor:
        """ Make predictions using the current model.

        :param uids: Batch of record unique identifiers.
        :param is_infer: Flag indicating whether to use the test data.

        :return: Model predictions.
        """
        logger.info(f"Member {self.id}: predicting")
        self.check_if_ready()
        _uid2tensor_idx = self.uid2tensor_idx_test if is_infer else self.uid2tensor_idx
        tensor_idx = [_uid2tensor_idx[uid] for uid in uids] if uids else None
        if is_infer:
            logger.info("Member %s: using test data" % self.id)
            if uids is None:
                X = self.device_dataset_test_split
            else:
                X = self.device_dataset_test_split[tensor_idx, :]
        else:
            X = self.device_dataset_train_split[tensor_idx, :]
        if is_infer:
            self._model.eval()
        predictions = self._model.predict(X)
        self._model.train()
        logger.debug(f"Member {self.id}: made predictions")
        return predictions

    def update_predict(self, upd: DataTensor, previous_batch: RecordsBatch, batch: RecordsBatch) -> DataTensor:
        """ Update model weights and make predictions.

        :param upd: Updated model weights.
        :param previous_batch: Previous batch of record unique identifiers.
        :param batch: Current batch of record unique identifiers.

        :return: Model predictions.
        """
        logger.info(f"Member {self.id}: updating and predicting")
        self.check_if_ready()
        if previous_batch is not None:
            self.update_weights(uids=previous_batch, upd=upd)

        predictions = self.predict(batch)
        self.iterations_counter += 1
        logger.debug(f"Member {self.id}: updated and predicted")
        return predictions
