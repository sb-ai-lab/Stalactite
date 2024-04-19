import logging
import time
from typing import List
from copy import copy

import mlflow
import torch
from sklearn.metrics import roc_auc_score, root_mean_squared_error

from stalactite.base import DataTensor, PartyCommunicator, IterationTime
from stalactite.communications.helpers import Method, MethodKwargs
from stalactite.ml.honest.base import Batcher
from stalactite.ml.honest.linear_regression.party_master import HonestPartyMasterLinReg

logger = logging.getLogger(__name__)


class HonestPartyMasterSplitNN(HonestPartyMasterLinReg):

    def predict(self, x: DataTensor, is_infer: bool = False, use_activation: bool = False) -> DataTensor:
        """ Make predictions using the master model.
        :return: Model predictions.
        """
        logger.info("Master: predicting.")
        self.check_if_ready()
        predictions = self._model.predict(x)
        if use_activation:
            predictions = self.activation(predictions)
        logger.info("Master: made predictions.")

        return predictions

    def update_weights(self, agg_members_output: DataTensor, upd: DataTensor) -> None:
        logger.info(f"Master: updating weights. Incoming tensor: {upd.size()}")
        self.check_if_ready()
        self._model.update_weights(x=agg_members_output, gradients=upd, is_single=False, optimizer=self._optimizer)
        logger.info("Master: successfully updated weights")

    def update_predict(self, upd: DataTensor, agg_members_output: DataTensor) -> DataTensor:
        logger.info("Master: updating and predicting.")
        self.check_if_ready()
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
            uids: List[str],
    ) -> List[DataTensor]:
        """ Compute updates for SplitNN.

        :param participating_members: List of participating party members identifiers.
        :param master_predictions: Master predictions.
        :param agg_predictions: Aggregated predictions.
        :param world_size: Number of party members.
        :param uids: Sub-iteration sequence number.

        :return: List of gradients as tensors.
        """
        logger.info("Master %s: computing updates (world size %s)" % (self.id, world_size))
        self.check_if_ready()
        self.iteration_counter += 1
        tensor_idx = [self.uid2tensor_idx[uid] for uid in uids]
        y = self.target[tensor_idx]
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

    def loop(self, batcher: Batcher, party: PartyCommunicator) -> None:
        """ Run main training loop on the VFL master.

        :param batcher: Batcher for creating training batches.
        :param party: Communicator instance used for communication between VFL agents.
        :return: None
        """
        logger.info("Master %s: entering training loop" % self.id)
        updates = self.make_init_updates(party.world_size)

        for titer in batcher:
            logger.debug(
                f"Master %s: train loop - starting batch %s (sub iter %s) on epoch %s"
                % (self.id, titer.seq_num, titer.subiter_seq_num, titer.epoch)
            )
            iter_start_time = time.time()
            # tasks for members
            update_predict_tasks = party.scatter(
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

            party_members_predictions = [
                task.result for task in party.gather(update_predict_tasks, recv_results=True)
            ]

            agg_members_predictions = self.aggregate(titer.participating_members, party_members_predictions)

            master_predictions = self.update_predict(
                upd=self.updates["master"], agg_members_output=agg_members_predictions)

            updates = self.compute_updates(
                titer.participating_members,
                master_predictions,
                agg_members_predictions,
                party.world_size,
                titer.batch,
            )

            if self.report_train_metrics_iteration > 0 and titer.seq_num % self.report_train_metrics_iteration == 0:
                logger.debug(
                    f"Master %s: train loop - reporting train metrics on iteration %s of epoch %s"
                    % (self.id, titer.seq_num, titer.epoch)
                )
                predict_tasks = party.broadcast(
                    Method.predict,
                    method_kwargs=MethodKwargs(other_kwargs={"uids": batcher.uids}),
                    participating_members=titer.participating_members,
                )

                party_members_predictions = [
                    task.result for task in party.gather(predict_tasks, recv_results=True)
                ]

                agg_members_predictions = self.aggregate(party.members, party_members_predictions, is_infer=True)
                master_predictions = self.predict(x=agg_members_predictions, use_activation=True)

                target = self.target[[self.uid2tensor_idx[uid] for uid in batcher.uids]]

                self.report_metrics(
                    target.numpy(), master_predictions.detach().numpy(), name="Train", step=titer.seq_num
                )

            if self.report_test_metrics_iteration > 0 and titer.seq_num % self.report_test_metrics_iteration == 0:
                logger.debug(
                    f"Master %s: train loop - reporting test metrics on iteration %s of epoch %s"
                    % (self.id, titer.seq_num, titer.epoch)
                )
                predict_test_tasks = party.broadcast(
                    Method.predict,
                    method_kwargs=MethodKwargs(other_kwargs={"uids": None, "is_infer": True}),
                    participating_members=titer.participating_members,
                )

                party_members_predictions = [
                    task.result for task in party.gather(predict_test_tasks, recv_results=True)
                ]
                agg_members_predictions = self.aggregate(party.members, party_members_predictions, is_infer=True)
                master_predictions = self.predict(x=agg_members_predictions, use_activation=True)
                self.report_metrics(
                    self.test_target.numpy(), master_predictions.detach().numpy(), name="Test", step=titer.seq_num
                )

            self.iteration_times.append(
                IterationTime(client_id=self.id, iteration=titer.seq_num, iteration_time=time.time() - iter_start_time)
            )

    def report_metrics(self, y: DataTensor, predictions: DataTensor, name: str, step: int) -> None:
        if self.binary:
            for avg in ["macro", "micro"]:
                try:
                    roc_auc = roc_auc_score(y, predictions, average=avg)
                except ValueError:
                    roc_auc = 0

                rmse = root_mean_squared_error(y, predictions)

                logger.info(f'{name} RMSE on step {step}: {rmse}')
                logger.info(f'{name} ROC AUC {avg} on step {step}: {roc_auc}')
                if self.run_mlflow:
                    mlflow.log_metric(f"{name.lower()}_roc_auc_{avg}", roc_auc, step=step)
                    mlflow.log_metric(f"{name.lower()}_rmse", rmse, step=step)
        else:

            avg = "macro"
            roc_auc = roc_auc_score(y, predictions, average=avg, multi_class="ovr")
            logger.info(f'{name} ROC AUC {avg} on step {step}: {roc_auc}')
            if self.run_mlflow:
                mlflow.log_metric(f"{name.lower()}_roc_auc_{avg}", roc_auc, step=step)
