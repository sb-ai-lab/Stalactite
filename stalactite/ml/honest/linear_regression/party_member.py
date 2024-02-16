import logging

from stalactite.base import RecordsBatch, DataTensor
from stalactite.ml.honest.base import HonestPartyMember

# TODO
from stalactite.models import LinearRegressionBatch

logger = logging.getLogger(__name__)

logger.setLevel(logging.DEBUG)
sh = logging.StreamHandler()
sh.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
logger.addHandler(sh)


class HonestPartyMemberLinReg(HonestPartyMember):
    def initialize_model(self) -> None:
        """ Initialize the model based on the specified model name. """
        self._model = LinearRegressionBatch(
            input_dim=self._dataset[self._data_params.train_split][self._data_params.features_key].shape[1],
            output_dim=1, reg_lambda=0.5
        )

    def update_weights(self, uids: RecordsBatch, upd: DataTensor) -> None:
        """ Update model weights based on input features and target values.

        :param uids: Batch of record unique identifiers.
        :param upd: Updated model weights.
        """
        logger.info("Member %s: updating weights. Incoming tensor: %s" % (self.id, tuple(upd.size())))
        self.check_if_ready()
        X_train = self._dataset[self._data_params.train_split][self._data_params.features_key][[int(x) for x in uids]]
        self._model.update_weights(X_train, upd)
        logger.info("Member %s: successfully updated weights" % self.id)

    def predict(self, uids: RecordsBatch, use_test: bool = False) -> DataTensor:
        """ Make predictions using the current model.

        :param uids: Batch of record unique identifiers.
        :param use_test: Flag indicating whether to use the test data.

        :return: Model predictions.
        """
        logger.info("Member %s: predicting. Batch size: %s" % (self.id, len(uids)))
        self.check_if_ready()
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
        self.check_if_ready()
        uids = previous_batch if previous_batch is not None else batch
        self.update_weights(uids=uids, upd=upd)
        predictions = self.predict(batch)
        self.iterations_counter += 1
        logger.info("Member %s: updated and predicted." % self.id)
        return predictions