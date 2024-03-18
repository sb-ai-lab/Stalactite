import time
from abc import ABC, abstractmethod
from collections import defaultdict
import logging
from typing import List, Optional, Tuple, Dict
from copy import copy

import datasets
import scipy as sp
import torch

from stalactite.batching import ListBatcher, ConsecutiveListBatcher
from stalactite.base import (
    PartyMember,
    PartyMaster,
    PartyCommunicator,
    Method,
    MethodKwargs,
    Task,
    DataTensor,
    RecordsBatch,
    Batcher, PartyDataTensor,
)
from stalactite.configs import DataConfig, CommonConfig

logger = logging.getLogger(__name__)

logger.setLevel(logging.DEBUG)
sh = logging.StreamHandler()
sh.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
logger.addHandler(sh)


class HonestPartyMaster(PartyMaster, ABC):
    """ Abstract base class for the honest master party in the VFL experiment. """

    @abstractmethod
    def make_init_updates(self, world_size: int) -> PartyDataTensor:
        """ Make initial updates for party members.

        :param world_size: Number of party members.

        :return: Initial updates as a list of tensors.
        """
        ...

    @abstractmethod
    def initialize_model(self, do_load_model: bool = False) -> None:
        """ Initialize the model based on the specified model name. """
        ...

    @abstractmethod
    def initialize_optimizer(self) -> None:
        """ Initialize the optimizer."""
        ...

    @abstractmethod
    def aggregate(
            self, participating_members: List[str], party_predictions: PartyDataTensor, infer: bool = False
    ) -> DataTensor:
        """ Aggregate members` predictions.

        :param participating_members: List of participating party member identifiers.
        :param party_predictions: List of party predictions.
        :param infer: Flag indicating whether to perform inference.

        :return: Aggregated predictions.
        """
        ...

    @abstractmethod
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
        ...

    def run(self, party: PartyCommunicator) -> None:
        """ Run the VFL experiment with the master party.

        Current method implements initialization of the master and members, launches the main training loop,
        and finalizes the experiment.

        :param party: Communicator instance used for communication between VFL agents.
        :return: None
        """
        logger.info("Running master %s" % self.id)
        if self.do_train:
            self.fit(party)

        if self.do_predict:
            self.inference(party)

    def fit(self, party: PartyCommunicator) -> None:
        # sync train part
        records_uids_tasks = party.broadcast(
            Method.records_uids,
            participating_members=party.members,
        )
        records_uids_results = party.gather(records_uids_tasks, recv_results=True)

        collected_uids_results = [task.result for task in records_uids_results]

        party.broadcast(
            Method.initialize,
            participating_members=party.members,
            method_kwargs=MethodKwargs(other_kwargs={'is_infer': False})
        )
        self.initialize(is_infer=False)

        uids = self.synchronize_uids(collected_uids_results, world_size=party.world_size)
        party.broadcast(
            Method.register_records_uids,
            method_kwargs=MethodKwargs(other_kwargs={"uids": uids}),
            participating_members=party.members,
        )

        # sync test part
        records_uids_tasks = party.broadcast(
            Method.records_uids,
            participating_members=party.members,
            method_kwargs=MethodKwargs(other_kwargs={'is_infer': True})
        )
        records_uids_results = party.gather(records_uids_tasks, recv_results=True)
        collected_uids_results = [task.result for task in records_uids_results]
        uids_test = self.synchronize_uids(collected_uids_results, world_size=party.world_size, is_test=True)
        party.broadcast(
            Method.register_records_uids,
            method_kwargs=MethodKwargs(other_kwargs={"uids": uids_test, "is_test": True}),
            participating_members=party.members,
        )

        self.loop(batcher=self.make_batcher(uids=uids, party_members=party.members), party=party)

        party.broadcast(
            Method.finalize,
            participating_members=party.members,
        )
        self.finalize()
        logger.info("Finished master %s" % self.id)

    def inference(self, party: PartyCommunicator) -> None:
        records_uids_tasks = party.broadcast(
            Method.records_uids,
            participating_members=party.members,
            method_kwargs=MethodKwargs(other_kwargs={'is_infer': True})
        )
        records_uids_results = party.gather(records_uids_tasks, recv_results=True)

        collected_uids_results = [task.result for task in records_uids_results]

        party.broadcast(
            Method.initialize,
            participating_members=party.members,
            method_kwargs=MethodKwargs(other_kwargs={'is_infer': True})

        )
        self.initialize(is_infer=True)

        uids = self.synchronize_uids(collected_uids_results, world_size=party.world_size, is_test=True)
        party.broadcast(
            Method.register_records_uids,
            method_kwargs=MethodKwargs(other_kwargs={"uids": uids, "is_test": True}),
            participating_members=party.members,
        )
        self.inference_loop(
            batcher=self.make_batcher(uids=uids, party_members=party.members, is_infer=True),
            party=party
        )

        party.broadcast(
            Method.finalize,
            participating_members=party.members,
            method_kwargs=MethodKwargs(other_kwargs={'is_infer': True})
        )
        self.finalize(is_infer=True)
        logger.info("Finished master %s" % self.id)

    def loop(self, batcher: Batcher, party: PartyCommunicator) -> None:
        """ Run main training loop on the VFL master.

        :param batcher: Batcher for creating training batches.
        :param party: Communicator instance used for communication between VFL agents.

        :return: None
        """
        logger.info("Master %s: entering training loop" % self.id)
        updates = self.make_init_updates(party.world_size)
        # a = self._dataset["train_train"]["SK_ID_CURR"]# todo: remove

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
                party_predictions = [task.result for task in party.gather(predict_tasks, recv_results=True)]

                predictions = self.aggregate(party.members, party_predictions, infer=True)
                target = self.target[[self._uid2tensor_idx[uid] for uid in batcher.uids]]
                self.report_metrics(target, predictions, name="Train", step=titer.seq_num)

            if self.report_test_metrics_iteration > 0 and titer.seq_num % self.report_test_metrics_iteration == 0:
                logger.debug(
                    f"Master %s: train loop - reporting test metrics on iteration %s of epoch %s"
                    % (self.id, titer.seq_num, titer.epoch)
                )
                predict_test_tasks = party.broadcast(
                    Method.predict,
                    method_kwargs=MethodKwargs(other_kwargs={"uids": self.inference_target_uids, "use_test": True}),
                    participating_members=titer.participating_members,
                )

                party_predictions_test = [
                    task.result for task in party.gather(predict_test_tasks, recv_results=True)
                ]

                predictions = self.aggregate(party.members, party_predictions_test, infer=True)
                test_target = self.test_target[[self._uid2tensor_idx_test[uid] for uid in self.inference_target_uids]]

                self.report_metrics(test_target, predictions, name="Test", step=titer.seq_num)

            self._iter_time.append((titer.seq_num, time.time() - iter_start_time))

    def inference_loop(self, batcher: Batcher, party: PartyCommunicator) -> None:
        """ Run main inference loop on the VFL master.

        :param batcher: Batcher for creating validation batches.
        :param party: Communicator instance used for communication between VFL agents.

        :return: None
        """
        logger.info("Master %s: entering inference loop" % self.id)
        party_predictions_test = defaultdict(list)
        for titer in batcher:
            if titer.last_batch:
                break
            logger.debug(
                f"Master %s: inference loop - starting batch %s (sub iter %s) on epoch %s"
                % (self.id, titer.seq_num, titer.subiter_seq_num, titer.epoch)
            )
            iter_start_time = time.time()
            predict_test_tasks = party.broadcast(
                Method.predict,
                method_kwargs=MethodKwargs(other_kwargs={"uids": titer.batch, "use_test": True}),
                participating_members=titer.participating_members,
            )
            for task in party.gather(predict_test_tasks, recv_results=True):
                party_predictions_test[task.from_id].append(task.result)
            self._iter_time.append((titer.seq_num, time.time() - iter_start_time))

        party_predictions_test = self._aggregate_batched_predictions(party.members, party_predictions_test)
        predictions = self.aggregate(party.members, party_predictions_test, infer=True)

        target = self.test_target[[self._uid2tensor_idx_test[uid] for uid in batcher.uids]]
        self.report_metrics(target, predictions, name="Test", step=0)

    def _aggregate_batched_predictions(
            self, party_members: List[str], batched_party_predictions: Dict[str, List[DataTensor]]
    ) -> PartyDataTensor:
        return [torch.vstack(batched_party_predictions[member]) for member in party_members]


class HonestPartyMember(PartyMember, ABC):
    """ Implementation class of the honest PartyMember used for local and distributed VFL training. """
    master_id: str
    _dataset: datasets.DatasetDict
    _data_params: DataConfig
    _common_params: CommonConfig

    def __init__(
            self,
            uid: str,
            epochs: int,
            batch_size: int,
            eval_batch_size: int,
            member_record_uids: List[str],
            member_inference_record_uids: List[str],
            model_name: str,
            report_train_metrics_iteration: int,
            report_test_metrics_iteration: int,
            processor=None,
            is_consequently: bool = False,
            members: Optional[list[str]] = None,
            model_path: Optional[str] = None,
            do_train: bool = True,
            do_predict: bool = False,
            do_save_model: bool = False,
            use_inner_join: bool = False,
            model_params: dict = None
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
        :param is_consequently: Flag indicating whether to use the consequent implementation (including the make_batcher).
        :param members: List of the members if the algorithm is consequent.
        """
        self.id = uid
        self.epochs = epochs
        self._batch_size = batch_size
        self._eval_batch_size = eval_batch_size
        self._uids = member_record_uids
        self._infer_uids = member_inference_record_uids
        self._uids_to_use: Optional[List[str]] = None
        self._uids_to_use_test: Optional[List[str]] = None
        self.is_initialized = False
        self.is_finalized = False
        self.iterations_counter = 0
        self._model_name = model_name
        self._model_params = model_params
        self.report_train_metrics_iteration = report_train_metrics_iteration
        self.report_test_metrics_iteration = report_test_metrics_iteration
        self.processor = processor
        self.is_consequently = is_consequently
        self.members = members
        self.model_path = model_path
        self.do_train = do_train
        self.do_predict = do_predict
        self.do_save_model = do_save_model
        self._optimizer = None
        self.use_inner_join = use_inner_join

        if self.is_consequently:
            if self.members is None:
                raise ValueError('If consequent algorithm is initialized, the members must be passed.')

        if do_save_model and model_path is None:
            raise ValueError('You must set the model path to save the model (`do_save_model` is True).')

    @abstractmethod
    def initialize_model(self, do_load_model: bool = False) -> None:
        """ Initialize the model based on the specified model name. """
        ...

    @abstractmethod
    def initialize_optimizer(self) -> None:
        """ Initialize the optimizer."""
        ...

    @abstractmethod
    def predict(self, uids: Optional[RecordsBatch], use_test: bool = False) -> DataTensor:
        """ Make predictions using the initialized model.

        :param uids: Batch of record unique identifiers.
        :param use_test: Flag indicating whether to use the test data.

        :return: Model predictions.
        """
        ...

    @abstractmethod
    def update_predict(self, upd: DataTensor, previous_batch: RecordsBatch, batch: RecordsBatch) -> DataTensor:
        """ Update model weights and make predictions.

        :param upd: Updated model weights.
        :param previous_batch: Previous batch of record unique identifiers.
        :param batch: Current batch of record unique identifiers.

        :return: Model predictions.
        """
        ...

    def run(self, party: PartyCommunicator):
        """ Run the VFL experiment with the member party.

        Current method implements initialization of the member, launches the main training loop,
        and finalizes the member.

        :param party: Communicator instance used for communication between VFL agents.
        :return: None
        """
        logger.info("Running member %s" % self.id)
        if self.do_train:
            self._run(party, is_infer=False)

        if self.do_predict:
            self._run(party, is_infer=True)

    def _run(self, party: PartyCommunicator, is_infer: bool = False):
        # sync train
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

        if not is_infer:
            # sync test
            synchronize_uids_task = party.recv(
                Task(method_name=Method.records_uids, from_id=self.master_id, to_id=self.id)
            )
            uids = self.execute_received_task(synchronize_uids_task)
            party.send(self.master_id, Method.records_uids, result=uids)
            register_records_uids_task = party.recv(
                Task(method_name=Method.register_records_uids, from_id=self.master_id, to_id=self.id)
            )
            self.execute_received_task(register_records_uids_task)
            self.loop(batcher=self.make_batcher(), party=party)
        else:
            self.inference_loop(batcher=self.make_batcher(is_infer=True), party=party)

        finalize_task = party.recv(Task(method_name=Method.finalize, from_id=self.master_id, to_id=self.id))
        self.execute_received_task(finalize_task)
        logger.info("Finished member %s" % self.id)


    def _predict_metrics_loop(self, party: PartyCommunicator):
        predict_task = party.recv(Task(method_name=Method.predict, from_id=self.master_id, to_id=self.id))
        predictions = self.execute_received_task(predict_task)
        party.send(self.master_id, Method.predict, result=predictions)

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
                self._predict_metrics_loop(party)

            if self.report_test_metrics_iteration > 0 and titer.seq_num % self.report_test_metrics_iteration == 0:
                logger.debug(
                    f"Member %s: train loop - calculating train metrics on iteration %s of epoch %s"
                    % (self.id, titer.seq_num, titer.epoch)
                )
                self._predict_metrics_loop(party)

            self._iter_time.append((titer.seq_num, time.time() - iter_start_time))

    def inference_loop(self, batcher: Batcher, party: PartyCommunicator):
        """ Run main inference loop on the VFL member.

        :param batcher: Batcher for creating training batches.
        :param party: Communicator instance used for communication between VFL agents.

        :return: None
        """
        logger.info("Member %s: entering training loop" % self.id)

        for titer in batcher:
            if titer.last_batch:
                break
            if titer.participating_members is not None:
                if self.id not in titer.participating_members:
                    logger.debug(f'Member {self.id} skipping {titer.seq_num}.')
                    continue

            predict_task = party.recv(Task(method_name=Method.predict, from_id=self.master_id, to_id=self.id))
            predictions = self.execute_received_task(predict_task)
            party.send(self.master_id, Method.predict, result=predictions)

    def make_batcher(
            self,
            uids: Optional[List[str]] = None,
            party_members: Optional[List[str]] = None,
            is_infer: bool = False,
    ) -> Batcher:
        epochs = 1 if is_infer else self.epochs
        batch_size = self._eval_batch_size if is_infer else self._batch_size
        uids_to_use = self._uids_to_use_test if is_infer else self._uids_to_use
        if uids_to_use is None:
            raise RuntimeError("Cannot create make_batcher, you must `register_records_uids` first.")
        return self._create_batcher(epochs=epochs, uids=uids_to_use, batch_size=batch_size)

    def _create_batcher(self, epochs: int, uids: List[str], batch_size: int) -> Batcher:
        """Create a make_batcher for training.

        :param epochs: Number of training epochs.
        :param uids: List of unique identifiers for dataset rows.
        :param batch_size: Size of the training batch.
        """
        logger.info("Member %s: making a make_batcher for uids" % self.id)
        self.check_if_ready()
        if not self.is_consequently:
            return ListBatcher(epochs=epochs, members=None, uids=uids, batch_size=batch_size)
        else:
            return ConsecutiveListBatcher(
                epochs=epochs, members=self.members, uids=uids, batch_size=batch_size
            )

    def records_uids(self, is_infer: bool = False) -> Tuple[List[str], bool]:
        """ Get the list of existing dataset unique identifiers.

        :return: List of unique identifiers.
        """
        logger.info("Member %s: reporting existing record uids" % self.id)
        if is_infer:
            return self._infer_uids, self.use_inner_join
        return self._uids, self.use_inner_join

    def register_records_uids(self, uids: List[str], is_test: bool = False) -> None:
        """ Register unique identifiers to be used.

        :param uids: List of unique identifiers.
        :return: None
        """
        logger.info("Member %s: registering %s uids to be used." % (self.id, len(uids)))

        if is_test:
            self._uids_to_use_test = uids
        else:
            self._uids_to_use = uids
        self.fillna(is_test=is_test)

    def fillna(self, is_test: bool = False) -> None:
        """ Fills missing values for member's dataset"""
        uids_to_use = self._uids_to_use_test if is_test else self._uids_to_use
        _uids = self._infer_uids if is_test else self._uids
        _uid2tensor_idx = self._uid2tensor_idx_test if is_test else self._uid2tensor_idx
        split = self._data_params.test_split if is_test else self._data_params.train_split

        uids_to_fill = list(set(uids_to_use) - set(_uids))
        if len(uids_to_fill) == 0:
            return

        logger.info(f"Member {self.id} has {len(uids_to_fill)} missing values : using fillna...")
        start_idx = max(_uid2tensor_idx.values()) + 1
        idx = start_idx
        for uid in uids_to_fill:
            _uid2tensor_idx[uid] = idx
            idx += 1

        fill_shape = self._dataset[split][self._data_params.features_key].shape[1]
        member_id = self.id.split("-")[-1]
        features = copy(self._dataset[split][self._data_params.features_key])
        features = torch.cat([features, torch.zeros((len(uids_to_fill), fill_shape))])
        has_features_column = torch.tensor([1.0 for _ in range(start_idx)] + [0.0 for _ in range(len(uids_to_fill))])
        features = torch.cat([features, torch.unsqueeze(has_features_column, 1)], dim=1)

        ds = datasets.Dataset.from_dict(
            {
                "user_id": list(_uid2tensor_idx.keys()),
                f"features_part_{member_id}": features,
            }
        )

        ds = ds.with_format("torch")
        self._dataset[split] = ds
        if not is_test:
            self.initialize_model()
            self.initialize_optimizer()

    def initialize(self, is_infer: bool = False):
        """ Initialize the party member. """

        logger.info("Member %s: initializing" % self.id)
        self._dataset = self.processor.fit_transform()
        self._uid2tensor_idx = {uid: i for i, uid in enumerate(self._uids)}
        self._uid2tensor_idx_test = {uid: i for i, uid in enumerate(self._infer_uids)}
        self._data_params = self.processor.data_params
        self._common_params = self.processor.common_params
        self.initialize_model(do_load_model=is_infer)
        self.initialize_optimizer()
        self.is_initialized = True
        logger.info("Member %s: has been initialized" % self.id)

    def finalize(self, is_infer: bool = False) -> None:
        """ Finalize the party member. """
        logger.info("Member %s: finalizing" % self.id)
        self.check_if_ready()
        if self.do_save_model and not is_infer:
            self.save_model()
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
