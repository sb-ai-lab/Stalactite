import logging
import time
from typing import List

import mlflow
import torch
from sklearn import metrics
from sklearn.metrics import roc_auc_score

from stalactite.base import Batcher, DataTensor, PartyDataTensor, PartyMaster, PartyCommunicator, Method, MethodKwargs
from stalactite.batching import ConsecutiveListBatcher, ListBatcher
from stalactite.metrics import ComputeAccuracy, ComputeAccuracy_numpy

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
sh = logging.StreamHandler()
sh.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
logger.addHandler(sh)


class PartyMasterImpl(PartyMaster):
    """ Implementation class of the PartyMaster used for local and distributed VFL training. """

    def __init__(
            self,
            uid: str,
            epochs: int,
            report_train_metrics_iteration: int,
            report_test_metrics_iteration: int,
            # target: DataTensor,
            # test_target: DataTensor,
            target_uids: List[str],
            batch_size: int,
            model_update_dim_size: int,
            processor=None,
            run_mlflow: bool = False,
    ) -> None:
        """ Initialize PartyMasterImpl.

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
        # self.target = target
        # self.test_target = test_target
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

    def initialize(self) -> None:
        """ Initialize the party master. """
        logger.info("Master %s: initializing" % self.id)
        ds = self.processor.fit_transform()
        self.target = ds[self.processor.data_params.train_split][self.processor.data_params.label_key]
        self.test_target = ds[self.processor.data_params.test_split][self.processor.data_params.label_key]
        self.class_weights = self.processor.get_class_weights() \
            if self.processor.common_params.use_class_weights else None
        self.is_initialized = True

    def make_batcher(self, uids: List[str], party_members: List[str]) -> Batcher:
        """ Make a batcher for training.

        :param uids: List of unique identifiers of dataset records.
        :param party_members: List of party members` identifiers.

        :return: Batcher instance.
        """
        logger.info("Master %s: making a batcher for uids %s" % (self.id, uids))
        self._check_if_ready()
        assert party_members is not None, "Master is trying to initialize batcher without members list"
        return ListBatcher(epochs=self.epochs, members=party_members, uids=uids, batch_size=self._batch_size)

    def make_init_updates(self, world_size: int) -> PartyDataTensor:
        """ Make initial updates for party members.

        :param world_size: Number of party members.

        :return: Initial updates as a list of tensors.
        """
        logger.info("Master %s: making init updates for %s members" % (self.id, world_size))
        self._check_if_ready()
        return [torch.rand(self._batch_size) for _ in range(world_size)]

    def report_metrics(self, y: DataTensor, predictions: DataTensor, name: str) -> None:
        """ Report metrics based on target values and predictions.

        :param y: Target values.
        :param predictions: Model predictions.
        :param name: Name of the dataset ("Train" or "Test").

        :return: None
        """
        logger.info(
            f"Master %s: reporting metrics. Y dim: {y.size()}. " f"Predictions size: {predictions.size()}" % self.id
        )
        mae = metrics.mean_absolute_error(y, predictions.detach())
        acc = ComputeAccuracy().compute(y, predictions.detach())
        logger.info(f"Master %s: %s metrics (MAE): {mae}" % (self.id, name))
        logger.info(f"Master %s: %s metrics (Accuracy): {acc}" % (self.id, name))

        if self.run_mlflow:
            step = self.iteration_counter
            mlflow.log_metric(f"{name.lower()}_mae", mae, step=step)
            mlflow.log_metric(f"{name.lower()}_acc", acc, step=step)

    def aggregate(
            self, participating_members: List[str], party_predictions: PartyDataTensor, infer=False
    ) -> DataTensor:
        """ Aggregate members` predictions.

        :param participating_members: List of participating party member identifiers.
        :param party_predictions: List of party predictions.
        :param infer: Flag indicating whether to perform inference.

        :return: Aggregated predictions.
        """
        logger.info("Master %s: aggregating party predictions (num predictions %s)" % (self.id, len(party_predictions)))
        self._check_if_ready()
        if not infer:
            for member_id, member_prediction in zip(participating_members, party_predictions):
                self.party_predictions[member_id] = member_prediction
            party_predictions = list(self.party_predictions.values())

        return torch.sum(torch.stack(party_predictions, dim=1), dim=1)

    def compute_updates(
            self,
            participating_members: List[str],
            predictions: DataTensor,
            party_predictions: PartyDataTensor,
            world_size: int,
            subiter_seq_num: int,
    ) -> List[DataTensor]:
        """ Compute updates based on members` predictions.

        :param participating_members: List of participating party member identifiers.
        :param predictions: Model predictions.
        :param party_predictions: List of party predictions.
        :param world_size: Number of party members.
        :param subiter_seq_num: Sub-iteration sequence number.

        :return: List of updates as tensors.
        """
        logger.info("Master %s: computing updates (world size %s)" % (self.id, world_size))
        self._check_if_ready()
        self.iteration_counter += 1
        y = self.target[self._batch_size * subiter_seq_num: self._batch_size * (subiter_seq_num + 1)]

        for member_id in participating_members:
            party_predictions_for_upd = [v for k, v in self.party_predictions.items() if k != member_id]
            if len(party_predictions_for_upd) == 0:
                party_predictions_for_upd = [torch.rand(predictions.size())]
            pred_for_member_upd = torch.mean(torch.stack(party_predictions_for_upd), dim=0)
            member_update = y - torch.reshape(pred_for_member_upd, (-1,))
            self.updates[member_id] = member_update

        return [self.updates[member_id] for member_id in participating_members]

    def finalize(self):
        """ Finalize the party master. """
        logger.info("Master %s: finalizing" % self.id)
        self._check_if_ready()
        self.is_finalized = True

    def _check_if_ready(self):
        """ Check if the party master is ready for operations.

        Raise a RuntimeError if experiment has not been initialized or has already finished.
        """
        if not self.is_initialized and not self.is_finalized:
            raise RuntimeError("The master has not been initialized")


class PartyMasterImplConsequently(PartyMasterImpl):
    """ Implementation class of the PartyMaster used for local and VFL training in a sequential manner. """

    def make_batcher(self, uids: List[str], party_members: List[str]) -> Batcher:
        """ Make a batcher for training in sequential order.

        :param uids: List of unique identifiers for dataset records.
        :param party_members: List of party member identifiers.

        :return: ConsecutiveListBatcher instance.
        """
        logger.info("Master %s: making a batcher for uids %s" % (self.id, uids))
        self._check_if_ready()
        return ConsecutiveListBatcher(epochs=self.epochs, members=party_members, uids=uids, batch_size=self._batch_size)


class PartyMasterImplLogreg(PartyMasterImpl):
    """ Implementation class of the VFL PartyMaster specific to the Logistic Regression algorithm. """

    def make_init_updates(self, world_size: int) -> PartyDataTensor:
        """ Make initial updates for logistic regression.

        :param world_size: Number of party members.
        :return: Initial updates as a list of zero tensors.
        """
        logger.info("Master %s: making init updates for %s members" % (self.id, world_size))
        self._check_if_ready()
        return [torch.zeros(self._batch_size, self.target.shape[1]) for _ in range(world_size)]

    def aggregate(
            self, participating_members: List[str], party_predictions: PartyDataTensor, infer=False
    ) -> DataTensor:
        """ Aggregate party predictions for logistic regression.

        :param participating_members: List of participating party member identifiers.
        :param party_predictions: List of party predictions.
        :param infer: Flag indicating whether to perform inference.

        :return: Aggregated predictions after applying sigmoid function.
        """
        logger.info("Master %s: aggregating party predictions (num predictions %s)" % (self.id, len(party_predictions)))
        self._check_if_ready()
        if not infer:
            for member_id, member_prediction in zip(participating_members, party_predictions):
                self.party_predictions[member_id] = member_prediction
            party_predictions = list(self.party_predictions.values())
            predictions = torch.sum(torch.stack(party_predictions, dim=1), dim=1)
        else:
            predictions = torch.sigmoid(torch.sum(torch.stack(party_predictions, dim=1), dim=1))
        return predictions

    def compute_updates(
            self,
            participating_members: List[str],
            predictions: DataTensor,
            party_predictions: PartyDataTensor,
            world_size: int,
            subiter_seq_num: int,
    ) -> List[DataTensor]:
        """ Compute updates for logistic regression.

         :param participating_members: List of participating party members identifiers.
         :param predictions: Model predictions.
         :param party_predictions: List of party predictions.
         :param world_size: Number of party members.
         :param subiter_seq_num: Sub-iteration sequence number.

         :return: List of gradients as tensors.
         """
        logger.info("Master %s: computing updates (world size %s)" % (self.id, world_size))
        self._check_if_ready()
        self.iteration_counter += 1
        y = self.target[self._batch_size * subiter_seq_num: self._batch_size * (subiter_seq_num + 1)]

        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=self.class_weights)
        loss = criterion(torch.squeeze(predictions), y.float())
        grads = torch.autograd.grad(outputs=loss, inputs=predictions)

        for i, member_id in enumerate(participating_members):
            self.updates[member_id] = grads[0]

        return [self.updates[member_id] for member_id in participating_members]

    def report_metrics(self, y: DataTensor, predictions: DataTensor, name: str) -> None:
        """Report metrics for logistic regression.

        Compute main classification metrics, if `use_mlflow` parameter was set to true, log them to MlFLow, log them to
        stdout.

        :param y: Target values.
        :param predictions: Model predictions.
        :param name: Name of the dataset ("Train" or "Test").
        """
        logger.info(
            f"Master %s: reporting metrics. Y dim: {y.size()}. " f"Predictions size: {predictions.size()}" % self.id
        )

        y = y.numpy()
        predictions = predictions.detach().numpy()

        mae = metrics.mean_absolute_error(y, predictions)
        acc = ComputeAccuracy_numpy(is_linreg=False).compute(y, predictions)
        logger.info(f"Master %s: %s metrics (MAE): {mae}" % (self.id, name))
        logger.info(f"Master %s: %s metrics (Accuracy): {acc}" % (self.id, name))
        step = self.iteration_counter
        if self.run_mlflow:
            mlflow.log_metric(f"{name.lower()}_mae", mae, step=step)
            mlflow.log_metric(f"{name.lower()}_acc", acc, step=step)

        for avg in ["macro", "micro"]:
            try:
                roc_auc = roc_auc_score(y, predictions, average=avg)
            except ValueError:
                roc_auc = 0
            logger.info(f'{name} ROC AUC {avg} on step {step}: {roc_auc}')
            if self.run_mlflow:
                mlflow.log_metric(f"{name.lower()}_roc_auc_{avg}", roc_auc, step=step)
