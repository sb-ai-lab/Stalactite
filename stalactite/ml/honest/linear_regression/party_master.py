from typing import Optional, Any
import logging
from typing import List

import mlflow
import torch
import torch.nn as nn
from sklearn import metrics

from stalactite.base import DataTensor, PartyDataTensor
from stalactite.batching import ConsecutiveListBatcher, ListBatcher
from stalactite.ml.honest.base import HonestPartyMaster, Batcher
from stalactite.metrics import ComputeAccuracy


logger = logging.getLogger(__name__)


class HonestPartyMasterLinReg(HonestPartyMaster):
    """ Implementation class of the PartyMaster used for local and distributed VFL training. """

    def __init__(
            self,
            uid: str,
            epochs: int,
            report_train_metrics_iteration: int,
            report_test_metrics_iteration: int,
            target_uids: List[str],
            inference_target_uids: List[str],
            batch_size: int,
            eval_batch_size: int,
            model_update_dim_size: int,
            processor=None,
            run_mlflow: bool = False,
            do_train: bool = True,
            do_predict: bool = False,
            model_name: str = None,
            model_params: dict = None,
            seed: int = None,
            device: str = 'cpu',
            model_path: Optional[str] = None,
            do_save_model: bool = False,
    ) -> None:
        """ Initialize PartyMaster.

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
        self.target_uids = target_uids
        self.inference_target_uids = inference_target_uids
        self.is_initialized = False
        self.is_finalized = False
        self._batch_size = batch_size
        self._eval_batch_size = eval_batch_size
        self._weights_dim = model_update_dim_size
        self.run_mlflow = run_mlflow
        self.processor = processor
        self.do_train = do_train
        self.do_predict = do_predict
        self.iteration_counter = 0
        self.party_predictions = dict()
        self.updates = {"master": torch.tensor([])}
        self._model_name = model_name
        self.aggregated_output = None
        self._model_params = model_params
        self.seed = seed
        self.device = torch.device(device)
        self.do_save_model = do_save_model
        self.model_path = model_path

        self.uid2tensor_idx = None
        self.uid2tensor_idx_test = None

    def initialize_model(self, do_load_model: bool = False):
        pass

    def initialize_optimizer(self) -> None:
        pass

    def initialize(self, is_infer: bool = False) -> None:
        """ Initialize the party master. """
        logger.info(f"Master {self.id}: initializing")
        ds = self.processor.fit_transform()
        self.target = ds[self.processor.data_params.train_split][self.processor.data_params.label_key]
        self.test_target = ds[self.processor.data_params.test_split][self.processor.data_params.label_key]
        if self.uid2tensor_idx is None:
            self.uid2tensor_idx = {uid: i for i, uid in enumerate(self.target_uids)}
        if self.uid2tensor_idx_test is None:
            self.uid2tensor_idx_test = {uid: i for i, uid in enumerate(self.inference_target_uids)}
        self.class_weights = self.processor.get_class_weights() \
            if self.processor.common_params.use_class_weights else None
        self._data_params = self.processor.data_params
        self._common_params = self.processor.common_params

        if torch.equal(torch.unique(self.target), torch.tensor([0, 1])) or torch.max(self.target).item() <= 1:
            self.activation = nn.Sigmoid()
            self.binary = True
        else:
            self.activation = nn.Softmax(dim=1)
            self.binary = False

        if self._model_name is not None:
            self.initialize_model(do_load_model=is_infer)
            self.initialize_optimizer()

        self.target = self.target.to(self.device)
        self.test_target = self.test_target.to(self.device)
        self.is_initialized = True
        self.is_finalized = False
        logger.info(f"Master {self.id}: has been initialized")

    def make_batcher(
            self,
            uids: Optional[List[str]] = None,
            party_members: Optional[List[str]] = None,
            is_infer: bool = False,
    ) -> Batcher:
        """ Make a make_batcher for training.

        :param uids: List of unique identifiers of dataset records.
        :param party_members: List of party members` identifiers.

        :return: Batcher instance.
        """
        self.check_if_ready()
        batch_size = self._eval_batch_size if is_infer else self._batch_size
        epochs = 1 if is_infer else self.epochs
        if uids is None:
            raise RuntimeError('Master must initialize batcher with collected uids.')
        logger.info(f"Master {self.id} makes a batcher for {len(uids)} uids")
        assert party_members is not None, "Master is trying to initialize make_batcher without members list"
        return ListBatcher(epochs=epochs, members=party_members, uids=uids, batch_size=batch_size)

    def make_init_updates(self, world_size: int) -> PartyDataTensor:
        """ Make initial updates for party members.

        :param world_size: Number of party members.

        :return: Initial updates as a list of tensors.
        """
        logger.info(f"Master {self.id}: makes initial updates for {world_size} members")
        self.check_if_ready()
        return [torch.zeros(self._batch_size, device=self.device) for _ in range(world_size)]

    def report_metrics(self, y: DataTensor, predictions: DataTensor, name: str, step: int) -> None:
        """ Report metrics based on target values and predictions.

        :param y: Target values.
        :param predictions: Model predictions.
        :param name: Name of the dataset ("Train" or "Test").

        :return: None
        """
        logger.info(f"Master {self.id} reporting metrics")
        logger.debug(f"Predictions size: {predictions.size()}, Target size: {y.size()}")
        postfix = '-infer' if step == -1 else ""
        step = step if step != -1 else None
        y = y.cpu()
        predictions = predictions.cpu()
        mae = metrics.mean_absolute_error(y, predictions)
        acc = ComputeAccuracy().compute(y, predictions)
        logger.info(f"{name} metrics (MAE): {mae}")
        logger.info(f"{name} metrics (Accuracy): {acc}")

        if self.run_mlflow:
            mlflow.log_metric(f"{name.lower()}_mae{postfix}", mae, step=step)
            mlflow.log_metric(f"{name.lower()}_acc{postfix}", acc, step=step)

    def aggregate(
            self, participating_members: List[str], party_predictions: PartyDataTensor, is_infer: bool = False
    ) -> DataTensor:
        """ Aggregate members` predictions.

        :param participating_members: List of participating party member identifiers.
        :param party_predictions: List of party predictions.
        :param infer: Flag indicating whether to perform inference.

        :return: Aggregated predictions.
        """
        logger.info(f"Master {self.id}: aggregates party predictions (number of predictions {len(party_predictions)})")
        self.check_if_ready()
        if not is_infer:
            for member_id, member_prediction in zip(participating_members, party_predictions):
                self.party_predictions[member_id] = member_prediction.to(self.device)
            party_predictions = list(self.party_predictions.values())

        return torch.sum(torch.stack(party_predictions, dim=1), dim=1)

    def compute_updates(
            self,
            participating_members: List[str],
            predictions: DataTensor,
            party_predictions: PartyDataTensor,
            world_size: int,
            uids: list[str],
    ) -> List[DataTensor]:
        """ Compute updates based on members` predictions.

        :param participating_members: List of participating party member identifiers.
        :param predictions: Model predictions.
        :param party_predictions: List of party predictions.
        :param world_size: Number of party members.
        :param uids: uids of record to use

        :return: List of updates as tensors.
        """
        logger.info(f"Master {self.id}: computes updates (world size {world_size})")
        self.check_if_ready()
        self.iteration_counter += 1
        tensor_idx = [self.uid2tensor_idx[uid] for uid in uids]
        y = self.target[tensor_idx]
        for member_id in participating_members:
            party_predictions_for_upd = [v for k, v in self.party_predictions.items() if k != member_id]
            if len(party_predictions_for_upd) == 0:
                party_predictions_for_upd = [torch.rand(predictions.size(), device=self.device)]
            pred_for_member_upd = torch.mean(torch.stack(party_predictions_for_upd), dim=0)
            member_update = y - torch.reshape(pred_for_member_upd, (-1,))
            self.updates[member_id] = member_update
        logger.debug(f"Master {self.id}: computed updates")
        return [self.updates[member_id] for member_id in participating_members]

    def finalize(self, is_infer: bool = False) -> None:
        """ Finalize the party master. """
        logger.info(f"Master {self.id}: finalizing")
        self.check_if_ready()
        self.is_finalized = True
        logger.info(f"Master {self.id}: has finalized")

    def check_if_ready(self):
        """ Check if the party master is ready for operations.

        Raise a RuntimeError if experiment has not been initialized or has already finished.
        """
        if not self.is_initialized and not self.is_finalized:
            raise RuntimeError("The master has not been initialized")

    def initialize_model_from_params(self, **model_params) -> Any:
        raise AttributeError('Honest master does not hold a model.')


class HonestPartyMasterLinRegConsequently(HonestPartyMasterLinReg):
    """ Implementation class of the PartyMaster used for local and VFL training in a sequential manner. """

    def make_batcher(
            self,
            uids: Optional[List[str]] = None,
            party_members: Optional[List[str]] = None,
            is_infer: bool = False,
    ) -> Batcher:
        """ Make a make_batcher for training in sequential order.

        :param uids: List of unique identifiers for dataset records.
        :param party_members: List of party member identifiers.

        :return: ConsecutiveListBatcher instance.
        """
        self.check_if_ready()
        epochs = 1 if is_infer else self.epochs
        batch_size = self._eval_batch_size if is_infer else self._batch_size
        if uids is None:
            raise RuntimeError('Master must initialize batcher with collected uids.')
        logger.info(f"Master {self.id} makes a batcher for {len(uids)} uids")
        if not is_infer:
            return ConsecutiveListBatcher(epochs=epochs, members=party_members, uids=uids, batch_size=batch_size)
        else:
            return ListBatcher(epochs=epochs, members=party_members, uids=uids, batch_size=batch_size)
