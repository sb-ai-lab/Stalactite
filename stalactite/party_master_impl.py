import logging
import time
from typing import List
from copy import copy

import mlflow
import torch
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from torch import nn
from torchsummary import summary

from stalactite.base import (Batcher, DataTensor, PartyDataTensor, PartyMaster, PartyCommunicator, Method, MethodKwargs)
from stalactite.batching import ConsecutiveListBatcher, ListBatcher
from stalactite.metrics import ComputeAccuracy, ComputeAccuracy_numpy
from stalactite.models.split_learning import EfficientNetTop, MLPTop, ResNetTop

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
            target_uids: List[str],
            batch_size: int,
            model_update_dim_size: int,
            processor=None,
            run_mlflow: bool = False,
            model_name: str = None,
            model_params: dict = None
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
        self.target_uids = target_uids
        self.is_initialized = False
        self.is_finalized = False
        self._batch_size = batch_size
        self._weights_dim = model_update_dim_size
        self.run_mlflow = run_mlflow
        self.processor = processor
        self.iteration_counter = 0
        self.party_predictions = dict()
        self.updates = {"master": torch.tensor([])}
        self._model_name = model_name
        self.aggregated_output = None
        self._model_params = model_params

    def initialize_model(self) -> None:

        """ Initialize the model based on the specified model name. """
        if self._model_name == "efficientnet":
            self._model = EfficientNetTop(**self._model_params)
            logger.info(summary(self._model, (self._model_params["input_dim"], 1, 1), device="cpu"))
        elif self._model_name == "mlp":
            self._model = MLPTop(**self._model_params)
        elif self._model_name == "resnet":
            self._model = ResNetTop(**self._model_params)
            logger.info(
                summary(self._model, (self._model_params["input_dim"],),
                        device="cpu", batch_size=self._batch_size)
            )
        else:
            raise ValueError("unknown model %s" % self._model_name)

        if self._model_name in ["mlp", "resnet"]:
            self._criterion = nn.BCEWithLogitsLoss(pos_weight=self.class_weights)
            self._activation = nn.Sigmoid()
        elif self._model_name in ["efficientnet"]:
            self._criterion = nn.CrossEntropyLoss(weight=self.class_weights)
            self._activation = nn.Softmax(dim=1)
        else:
            raise ValueError("unknown model %s" % self._model_name)

        if self.run_mlflow:
            mlflow.log_param("init_weights", self._model_params.get("init_weights", None))

    def initialize_optimizer(self) -> None:
        self._optimizer = torch.optim.SGD([
            {"params": self._model.parameters()},
        ],
            lr=self._common_params.learning_rate,
            momentum=self._common_params.momentum
        )

    def initialize(self) -> None:
        """ Initialize the party master. """
        logger.info("Master %s: initializing" % self.id)
        ds = self.processor.fit_transform()
        self.target = ds[self.processor.data_params.train_split][self.processor.data_params.label_key]
        self.test_target = ds[self.processor.data_params.test_split][self.processor.data_params.label_key]
        self.class_weights = self.processor.get_class_weights() \
            if self.processor.common_params.use_class_weights else None
        self._data_params = self.processor.data_params
        self._common_params = self.processor.common_params
        if self._model_name is not None:
            self.initialize_model()
            self.initialize_optimizer()
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
        return [torch.zeros(self._batch_size) for _ in range(world_size)]

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

    def finalize(self) -> None:
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
        return [torch.zeros(self._batch_size, self.target.shape[1] if self.processor.multilabel else 1)
                for _ in range(world_size)]

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

        :return: None.
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


class PartyMasterImplSplitNN(PartyMasterImpl):

    def predict(self, x: DataTensor, use_test: bool = False, use_activation: bool = False) -> DataTensor:
        """ Make predictions using the master model.
        :return: Model predictions.
        """
        logger.info("Master: predicting.")
        self._check_if_ready()
        predictions = self._model.predict(x)
        if use_activation:
            predictions = self._activation(predictions)
        logger.info("Master: made predictions.")

        return predictions

    def update_weights(self, agg_members_output: DataTensor, upd: DataTensor) -> None:
        logger.info(f"Master: updating weights. Incoming tensor: {upd.size()}")
        self._check_if_ready()
        self._model.update_weights(x=agg_members_output, gradients=upd, is_single=False, optimizer=self._optimizer)
        logger.info("Master: successfully updated weights")

    def update_predict(self, upd: DataTensor, agg_members_output: DataTensor) -> DataTensor:
        logger.info("Master: updating and predicting.")
        self._check_if_ready()
        # get aggregated output from previous batch if exist (we do not make update_weights if it's the first iter)
        if self.aggregated_output is not None:
            self.update_weights(
                agg_members_output=self.aggregated_output, upd=upd)
        predictions = self.predict(agg_members_output, use_activation=False)
        logger.info("Master: updated and predicted.")
        # save current agg_members_output for making update_predict for next batch
        self.aggregated_output = copy(agg_members_output)
        return predictions

    def compute_updates(
            self,
            participating_members: List[str],
            master_predictions: DataTensor,
            agg_predictions: DataTensor,
            world_size: int,
            subiter_seq_num: int,
    ) -> List[DataTensor]:
        """ Compute updates for SplitNN.

        :param participating_members: List of participating party members identifiers.
        :param master_predictions: Master predictions.
        :param agg_predictions: Aggregated predictions.
        :param world_size: Number of party members.
        :param subiter_seq_num: Sub-iteration sequence number.

        :return: List of gradients as tensors.
        """
        logger.info("Master %s: computing updates (world size %s)" % (self.id, world_size))
        self._check_if_ready()
        self.iteration_counter += 1
        y = self.target[self._batch_size * subiter_seq_num: self._batch_size * (subiter_seq_num + 1)]
        targets_type = torch.LongTensor if isinstance(self._criterion, torch.nn.CrossEntropyLoss) else torch.FloatTensor
        loss = self._criterion(torch.squeeze(master_predictions), y.type(targets_type))
        if self.run_mlflow:
            mlflow.log_metric("loss", loss.item(), step=self.iteration_counter)

        for i, member_id in enumerate(participating_members):
            self.updates[member_id] = torch.autograd.grad(
                outputs=loss, inputs=self.party_predictions[member_id], retain_graph=True
            )[0]
        self.updates["master"] = torch.autograd.grad(
            outputs=loss, inputs=master_predictions, retain_graph=True
        )[0]

        return [self.updates[member_id] for member_id in participating_members]

    def loop(self, batcher: Batcher, communicator: PartyCommunicator) -> None:
        """ Run main training loop on the VFL master.

        :param batcher: Batcher for creating training batches.
        :param communicator: Communicator instance used for communication between VFL agents.
        :return: None
        """
        logger.info("Master %s: entering training loop" % self.id)
        updates = self.make_init_updates(communicator.world_size)

        for titer in batcher:
            logger.debug(
                f"Master %s: train loop - starting batch %s (sub iter %s) on epoch %s"
                % (self.id, titer.seq_num, titer.subiter_seq_num, titer.epoch)
            )
            iter_start_time = time.time()
            # tasks for members
            update_predict_tasks = communicator.scatter(
                Method.update_predict,
                method_kwargs=[
                    MethodKwargs(
                        tensor_kwargs={"upd": participant_updates},
                        other_kwargs={"previous_batch": titer.previous_batch, "batch": titer.batch},
                    )
                    for participant_updates in updates
                ],
                participating_members=titer.participating_members,
            )

            ordered_gather = sorted(communicator.gather(update_predict_tasks, recv_results=True),
                                    key=lambda x: int(x.from_id.split('-')[-1]))

            party_members_predictions = [
                task.result for task in ordered_gather
            ]

            agg_members_predictions = self.aggregate(titer.participating_members, party_members_predictions)

            master_predictions = self.update_predict(
                upd=self.updates["master"], agg_members_output=agg_members_predictions)

            updates = self.compute_updates(
                titer.participating_members,
                master_predictions,
                agg_members_predictions,
                communicator.world_size,
                titer.subiter_seq_num,
            )

            if self.report_train_metrics_iteration > 0 and titer.seq_num % self.report_train_metrics_iteration == 0:
                logger.debug(
                    f"Master %s: train loop - reporting train metrics on iteration %s of epoch %s"
                    % (self.id, titer.seq_num, titer.epoch)
                )
                predict_tasks = communicator.broadcast(
                    Method.predict,
                    method_kwargs=MethodKwargs(other_kwargs={"uids": batcher.uids}),
                    participating_members=titer.participating_members,
                )

                ordered_gather = sorted(communicator.gather(predict_tasks, recv_results=True),
                                        key=lambda x: int(x.from_id.split('-')[-1]))
                party_members_predictions = [task.result for task in ordered_gather]

                agg_members_predictions = self.aggregate(communicator.members, party_members_predictions, infer=True)
                master_predictions = self.predict(x=agg_members_predictions, use_activation=True)

                self.report_metrics(self.target.numpy(), master_predictions.detach().numpy(), name="Train")

            if self.report_test_metrics_iteration > 0 and titer.seq_num % self.report_test_metrics_iteration == 0:
                logger.debug(
                    f"Master %s: train loop - reporting test metrics on iteration %s of epoch %s"
                    % (self.id, titer.seq_num, titer.epoch)
                )
                predict_test_tasks = communicator.broadcast(
                    Method.predict,
                    method_kwargs=MethodKwargs(other_kwargs={"uids": batcher.uids, "use_test": True}),
                    participating_members=titer.participating_members,
                )
                ordered_gather = sorted(communicator.gather(predict_test_tasks, recv_results=True),
                                        key=lambda x: int(x.from_id.split('-')[-1]))

                party_members_predictions = [task.result for task in ordered_gather]
                agg_members_predictions = self.aggregate(communicator.members, party_members_predictions, infer=True)
                master_predictions = self.predict(x=agg_members_predictions, use_activation=True)
                self.report_metrics(self.test_target.numpy(), master_predictions.detach().numpy(), name="Test")

            self._iter_time.append((titer.seq_num, time.time() - iter_start_time))

    def report_metrics(self, y: DataTensor, predictions: DataTensor, name: str) -> None:
        step = self.iteration_counter
        for avg in ["macro", "micro"]:
            try:
                roc_auc = roc_auc_score(y, predictions, average=avg)
            except ValueError:
                roc_auc = 0
            logger.info(f'{name} ROC AUC {avg} on step {step}: {roc_auc}')
            if self.run_mlflow:
                mlflow.log_metric(f"{name.lower()}_roc_auc_{avg}", roc_auc, step=step)


class PartyMasterImplEfficientNetSplitNN(PartyMasterImplSplitNN):

    def aggregate(
            self, participating_members: List[str], party_predictions: PartyDataTensor, infer=False
    ) -> DataTensor:
        logger.info("Master %s: aggregating party predictions (num predictions %s)" % (self.id, len(party_predictions)))
        self._check_if_ready()

        for member_id, member_prediction in zip(participating_members, party_predictions):
            self.party_predictions[member_id] = member_prediction
        party_predictions = list(self.party_predictions.values())
        predictions = torch.mean(torch.stack(party_predictions, dim=1), dim=1)
        return predictions

    def report_metrics(self, y: DataTensor, predictions: DataTensor, name: str) -> None:
        logger.info(
            f"Master : reporting metrics. Y dim: {y.size}. Predictions size: {predictions.size}"
        )
        step = self.iteration_counter
        avg = "macro"
        roc_auc = roc_auc_score(y, predictions, average=avg, multi_class="ovr")
        logger.info(f'{name} roc_auc_{avg} on step {step}: {roc_auc}')
        if self.run_mlflow:
            mlflow.log_metric(f"{name.lower()}_roc_auc_{avg}", roc_auc, step=step)


class PartyMasterImplMLPSplitNN(PartyMasterImplSplitNN):

    def aggregate(
            self, participating_members: List[str], party_predictions: PartyDataTensor, infer=False
    ) -> DataTensor:
        logger.info("Master %s: aggregating party predictions (num predictions %s)" % (self.id, len(party_predictions)))
        self._check_if_ready()

        for member_id, member_prediction in zip(participating_members, party_predictions):
            self.party_predictions[member_id] = member_prediction
        party_predictions = list(self.party_predictions.values())
        predictions = torch.sum(torch.stack(party_predictions, dim=1), dim=1)
        return predictions


class PartyMasterImplResNetSplitNN(PartyMasterImplSplitNN):

    def aggregate(
            self, participating_members: List[str], party_predictions: PartyDataTensor, infer=False
    ) -> DataTensor:
        logger.info("Master %s: aggregating party predictions (num predictions %s)" % (self.id, len(party_predictions)))
        self._check_if_ready()

        for member_id, member_prediction in zip(participating_members, party_predictions):
            self.party_predictions[member_id] = member_prediction
        party_predictions = list(self.party_predictions.values())
        predictions = torch.cat(party_predictions, dim=1)
        return predictions
