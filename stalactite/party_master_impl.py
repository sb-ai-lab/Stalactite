import logging
import uuid
from typing import List, Optional

import torch
import mlflow
from sklearn import metrics

from stalactite.base import PartyMaster, DataTensor, Batcher, PartyDataTensor, PartyMember, Party
from stalactite.metrics import ComputeAccuracy
from stalactite.batching import ListBatcher

logger = logging.getLogger(__name__)


class PartyMasterImpl(PartyMaster):
    def __init__(self,
                 uid: str,
                 epochs: int,
                 report_train_metrics_iteration: int,
                 report_test_metrics_iteration: int,
                 target: DataTensor,
                 test_target: DataTensor,
                 target_uids: List[str],
                 batch_size: int,
                 model_update_dim_size: int,
                 run_mlflow: bool = False):
        self.id = uid
        self.epochs = epochs
        self.report_train_metrics_iteration = report_train_metrics_iteration
        self.report_test_metrics_iteration = report_test_metrics_iteration
        self.target = target
        self.test_target =test_target
        self.target_uids = target_uids
        self.is_initialized = False
        self.is_finalized = False
        self._batch_size = batch_size
        self._weights_dim = model_update_dim_size
        self.iteration_counter = 0
        self.run_mlflow = run_mlflow

    def master_initialize(self, party: Party):
        logger.info("Master %s: initializing" % self.id)
        party.initialize()
        self.is_initialized = True

    def make_batcher(self, uids: List[str], party: Party) -> Batcher:
        logger.info("Master %s: making a batcher for uids %s" % (self.id, uids))
        self._check_if_ready()
        return ListBatcher(epochs=self.epochs, members=party.members, uids=uids, batch_size=self._batch_size)

    def make_init_updates(self, world_size: int) -> PartyDataTensor:
        logger.info("Master %s: making init updates for %s members" % (self.id, world_size))
        self._check_if_ready()
        return [torch.rand(self._batch_size) for _ in range(world_size)]

    def report_metrics(self, y: DataTensor, predictions: DataTensor, name: str):
            logger.info(f"Master %s: reporting metrics. Y dim: {y.size()}. "
                        f"Predictions size: {predictions.size()}" % self.id)
            mae = metrics.mean_absolute_error(y, predictions)
            acc = ComputeAccuracy().compute(y, predictions)
            logger.info(f"Master %s: %s metrics (MAE): {mae}" % (self.id, name))
            logger.info(f"Master %s: %s metrics (Accuracy): {acc}" % (self.id, name))

            if self.run_mlflow:
                step = self.iteration_counter
                mlflow.log_metric(f"{name.lower()}_mae", mae, step=step)
                mlflow.log_metric(f"{name.lower()}_acc", acc, step=step)

    def aggregate(self, participating_members: List[str], party_predictions: PartyDataTensor) -> DataTensor:
        logger.info("Master %s: aggregating party predictions (num predictions %s)" % (self.id, len(party_predictions)))
        self._check_if_ready()
        return torch.sum(torch.stack(party_predictions, dim=1), dim=1)

    def compute_updates(self,
                        participating_members: List[str],
                        predictions: DataTensor,
                        party_predictions: PartyDataTensor,
                        world_size: int,
                        subiter_seq_num: int) -> List[DataTensor]:

        # :(self, predictions: DataTensor, party_predictions: PartyDataTensor,
        #                 world_size: int, iter_in_batch:int) -> List[DataTensor]:
        logger.info("Master %s: computing updates (world size %s)" % (self.id, world_size))
        self._check_if_ready()
        self.iteration_counter += 1
        # y = self.target[self._batch_size*(self.iteration_counter-1):self._batch_size*self.iteration_counter]
        y = self.target[self._batch_size*subiter_seq_num:self._batch_size*(subiter_seq_num+1)] #todo: make it in batcher

        updates = []
        participating_members = [int(p.split('-')[-1]) for p in participating_members]
        for member_id in participating_members:
            party_predictions_for_upd = [p for i, p in enumerate(party_predictions) if i != member_id]
            member_pred = torch.mean(torch.stack(party_predictions_for_upd), dim=0) #todo: for debug
            member_update = y - torch.reshape(member_pred, (-1,))
            updates.append(member_update)
        return updates

    def master_finalize(self, party: Party):
        logger.info("Master %s: finalizing" % self.id)
        self._check_if_ready()
        party.finalize()
        self.is_finalized = True

    def _check_if_ready(self):
        if not self.is_initialized and not self.is_finalized:
            raise RuntimeError("The member has not been initialized")


class PartyMasterImplConsequently(PartyMasterImpl):
    def loop(self, batcher: Batcher, party: Party):
        logger.info("Master %s: entering training loop" % self.id)
        updates = self.make_init_updates(party.world_size)

        for titer in batcher:
            logger.debug(f"Master %s: train loop - starting batch %s (sub iter %s) on epoch %s"
                         % (self.id, titer.seq_num, titer.subiter_seq_num, titer.epoch))

            if titer.seq_num == 0:
                logger.info("making first update")
                party_predictions = party.update_predict(
                    titer.participating_members, titer.batch, titer.previous_batch, updates
                )
                predictions = self.aggregate(titer.participating_members, party_predictions)
                # updates = self.compute_updates(
                #     titer.participating_members, predictions, party_predictions, party.world_size, titer.subiter_seq_num
                # )

            members_to_update = titer.participating_members
            for member_name in members_to_update:
                member_id = int(member_name.split("-")[-1])
                # predict from one member
                member_predictions = party.update_predict(
                    [member_name], titer.batch, titer.previous_batch, [updates[member_id]]
                )
                party_predictions[member_id] =member_predictions[0]
                # useless
                predictions = self.aggregate(titer.participating_members, party_predictions)
                member_updates = self.compute_updates(
                    [member_name], predictions, party_predictions, party.world_size, titer.subiter_seq_num
                )
                updates[member_id] = member_updates[0]

                if self.report_train_metrics_iteration > 0 and titer.seq_num % self.report_train_metrics_iteration == 0:
                    logger.debug(f"Master %s: train loop - reporting train metrics on iteration %s of epoch %s"
                                 % (self.id, titer.seq_num, titer.epoch))
                    party_predictions_for_metrics = party.predict(batcher.uids)
                    predictions = self.aggregate(party.members, party_predictions_for_metrics)
                    self.report_metrics(self.target, predictions, name="Train")

                if self.report_test_metrics_iteration > 0 and titer.seq_num % self.report_test_metrics_iteration == 0:
                    logger.debug(f"Master %s: train loop - reporting test metrics on iteration %s of epoch %s"
                                 % (self.id, titer.seq_num, titer.epoch))
                    party_predictions_for_metrics = party.predict(uids=batcher.uids, use_test=True)
                    predictions = self.aggregate(party.members, party_predictions_for_metrics)
                    self.report_metrics(self.test_target, predictions, name="Test")
