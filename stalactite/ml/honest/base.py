import time
from abc import ABC, abstractmethod
from concurrent.futures import Future
from dataclasses import dataclass, field
import logging
from typing import List, Optional, Tuple, Any, Iterator, Union

import scipy as sp
import torch

from stalactite.batching import ListBatcher, ConsecutiveListBatcher
from stalactite.models import LinearRegressionBatch, LogisticRegressionBatch
from stalactite.base import (
    PartyMember,
    PartyMaster,
    PartyCommunicator,
    Method,
    MethodKwargs,
    Task,
    DataTensor,
    RecordsBatch,
    Batcher,
)

logger = logging.getLogger(__name__)

logger.setLevel(logging.DEBUG)
sh = logging.StreamHandler()
sh.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
logger.addHandler(sh)


class HonestPartyMaster(PartyMaster, ABC):
    """ Abstract base class for the honest master party in the VFL experiment. """

    def run(self, party: PartyCommunicator) -> None:
        """ Run the VFL experiment with the master party.

        Current method implements initialization of the master and members, launches the main training loop,
        and finalizes the experiment.

        :param party: Communicator instance used for communication between VFL agents.
        :return: None
        """
        logger.info("Running master %s" % self.id)

        records_uids_tasks = party.broadcast(
            Method.records_uids,
            participating_members=party.members,
        )

        records_uids_results = party.gather(records_uids_tasks, recv_results=True)

        collected_uids_results = [task.result for task in records_uids_results]

        party.broadcast(
            Method.initialize,
            participating_members=party.members,
        )
        self.initialize()

        uids = self.synchronize_uids(collected_uids_results, world_size=party.world_size)
        party.broadcast(
            Method.register_records_uids,
            method_kwargs=MethodKwargs(other_kwargs={"uids": uids}),
            participating_members=party.members,
        )

        self.loop(batcher=self.make_batcher(uids=uids, party_members=party.members), party=party)

        party.broadcast(
            Method.finalize,
            participating_members=party.members,
        )
        self.finalize()
        logger.info("Finished master %s" % self.id)

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
            if titer.seq_num == 0:
                updates = updates[:len(titer.participating_members)]
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

            party_predictions = [task.result for task in party.gather(update_predict_tasks, recv_results=True)]

            predictions = self.aggregate(titer.participating_members, party_predictions)

            updates = self.compute_updates(
                titer.participating_members,
                predictions,
                party_predictions,
                party.world_size,
                titer.subiter_seq_num,
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
                party_predictions = [task.result for task in party.gather(predict_tasks, recv_results=True)]

                predictions = self.aggregate(party.members, party_predictions, infer=True)
                self.report_metrics(self.target, predictions, name="Train")

            if self.report_test_metrics_iteration > 0 and titer.seq_num % self.report_test_metrics_iteration == 0:
                logger.debug(
                    f"Master %s: train loop - reporting test metrics on iteration %s of epoch %s"
                    % (self.id, titer.seq_num, titer.epoch)
                )
                predict_test_tasks = party.broadcast(
                    Method.predict,
                    method_kwargs=MethodKwargs(other_kwargs={"uids": batcher.uids, "use_test": True}),
                    participating_members=titer.participating_members,
                )

                party_predictions_test = [
                    task.result for task in party.gather(predict_test_tasks, recv_results=True)
                ]

                predictions = self.aggregate(party.members, party_predictions_test, infer=True)
                self.report_metrics(self.test_target, predictions, name="Test")

            self._iter_time.append((titer.seq_num, time.time() - iter_start_time))


class HonestPartyMember(PartyMember, ABC):
    """ Implementation class of the honest PartyMember used for local and distributed VFL training. """

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
        else:
            raise ValueError("unknown model %s" % self._model_name)

    def _check_if_ready(self):
        """ Check if the party member is ready for operations.

        Raise a RuntimeError if experiment has not been initialized or has already finished.
        """
        if not self.is_initialized and not self.is_finalized:
            raise RuntimeError("The member has not been initialized")

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

    def initialize(self) -> None:
        """ Initialize the party member. """

        logger.info("Member %s: initializing" % self.id)
        self._dataset = self.processor.fit_transform()
        self._data_params = self.processor.data_params
        self._common_params = self.processor.common_params
        self.initialize_model()
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
        self._model.update_weights(X_train, upd)
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

    # def _execute_received_task(self, task: Task) -> Optional[Union[DataTensor, List[str]]]:
    #     """ Execute received method on a member.
    #
    #     :param task: Received task to execute.
    #     :return: Execution result.
    #     """
    #     return getattr(self, task.method_name)(**task.kwargs_dict)

    def run(self, party: PartyCommunicator):
        """ Run the VFL experiment with the member party.

        Current method implements initialization of the member, launches the main training loop,
        and finalizes the member.

        :param party: Communicator instance used for communication between VFL agents.
        :return: None
        """
        logger.info("Running member %s" % self.id)

        synchronize_uids_task = party.recv(
            Task(method_name=Method.records_uids, from_id=self.master_id, to_id=self.id)
        )
        uids = self.execute_received_task(synchronize_uids_task)
        party.send(self.master_id, Method.records_uids, result=uids)

        initialize_task = party.recv(Task(method_name=Method.initialize, from_id=self.master_id, to_id=self.id))
        self.execute_received_task(initialize_task)

        register_records_uids_task = party.recv(
            Task(method_name=Method.register_records_uids, from_id=self.master_id, to_id=self.id)
        )
        self.execute_received_task(register_records_uids_task)

        self.loop(batcher=self.batcher, party=party)

        finalize_task = party.recv(Task(method_name=Method.finalize, from_id=self.master_id, to_id=self.id))
        self.execute_received_task(finalize_task)
        logger.info("Finished member %s" % self.id)

    def loop(self, batcher: Batcher, party: PartyCommunicator):
        """ Run main training loop on the VFL member.

        :param batcher: Batcher for creating training batches.
        :param party: Communicator instance used for communication between VFL agents.

        :return: None
        """
        logger.info("Member %s: entering training loop" % self.id)

        for titer in batcher:
            if titer.participating_members is not None:
                if self.id not in titer.participating_members:
                    logger.debug(f'Member {self.id} skipping {titer.seq_num}.')
                    continue
            logger.debug(
                f"Member %s: train loop - starting batch %s (sub iter %s) on epoch %s"
                % (self.id, titer.seq_num, titer.subiter_seq_num, titer.epoch)
            )

            iter_start_time = time.time()
            update_predict_task = party.recv(
                Task(method_name=Method.update_predict, from_id=self.master_id, to_id=self.id)
            )
            predictions = self.execute_received_task(update_predict_task)
            party.send(self.master_id, Method.update_predict, result=predictions)

            if self.report_train_metrics_iteration > 0 and titer.seq_num % self.report_train_metrics_iteration == 0:
                logger.debug(
                    f"Member %s: train loop - calculating train metrics on iteration %s of epoch %s"
                    % (self.id, titer.seq_num, titer.epoch)
                )
                predict_task = party.recv(Task(method_name=Method.predict, from_id=self.master_id, to_id=self.id))
                predictions = self.execute_received_task(predict_task)
                party.send(self.master_id, Method.predict, result=predictions)

            if self.report_test_metrics_iteration > 0 and titer.seq_num % self.report_test_metrics_iteration == 0:
                logger.debug(
                    f"Member %s: train loop - calculating train metrics on iteration %s of epoch %s"
                    % (self.id, titer.seq_num, titer.epoch)
                )
                predict_task = party.recv(Task(method_name=Method.predict, from_id=self.master_id, to_id=self.id))
                predictions = self.execute_received_task(predict_task)
                party.send(self.master_id, Method.predict, result=predictions)

            self._iter_time.append((titer.seq_num, time.time() - iter_start_time))
