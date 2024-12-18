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
    Task,
    DataTensor,
    RecordsBatch,
    Batcher,
    PartyDataTensor,
    IterationTime,
)
from stalactite.configs import DataConfig, CommonConfig
from stalactite.communications.helpers import Method, MethodKwargs

logger = logging.getLogger(__name__)


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
            self, participating_members: List[str], party_predictions: PartyDataTensor, is_infer: bool = False
    ) -> DataTensor:
        """ Aggregate members` predictions.

        :param participating_members: List of participating party member identifiers.
        :param party_predictions: List of party predictions.
        :param is_infer: Flag indicating whether to perform inference.

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
        logger.info(f"Running master {self.id}")
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
        uids_test = self.synchronize_uids(collected_uids_results, world_size=party.world_size, is_infer=True)
        party.broadcast(
            Method.register_records_uids,
            method_kwargs=MethodKwargs(other_kwargs={"uids": uids_test, "is_infer": True}),
            participating_members=party.members,
        )

        self.loop(batcher=self.make_batcher(uids=uids, party_members=party.members), party=party)

        party.broadcast(
            Method.finalize,
            participating_members=party.members,
        )
        self.finalize()
        logger.info(f"Finished master {self.id}")

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

        uids = self.synchronize_uids(collected_uids_results, world_size=party.world_size, is_infer=True)
        party.broadcast(
            Method.register_records_uids,
            method_kwargs=MethodKwargs(other_kwargs={"uids": uids, "is_infer": True}),
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
        logger.info(f"Finished master {self.id}")

    def loop(self, batcher: Batcher, party: PartyCommunicator) -> None:
        """ Run main training loop on the VFL master.

        :param batcher: Batcher for creating training batches.
        :param party: Communicator instance used for communication between VFL agents.

        :return: None
        """
        logger.info(f"Master {self.id}: entering training loop")
        updates = self.make_init_updates(party.world_size)
        for titer in batcher:
            logger.info(
                f"Master {self.id}: train loop - starting batch {titer.seq_num} (sub iter {titer.subiter_seq_num}) "
                f"on epoch {titer.epoch}"
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
                    f"Master {self.id}: train loop - reporting train metrics on iteration {titer.seq_num} "
                    f"of epoch {titer.epoch}"
                )
                predict_tasks = party.broadcast(
                    Method.predict,
                    method_kwargs=MethodKwargs(other_kwargs={"uids": batcher.uids}),
                    participating_members=titer.participating_members,
                )
                party_predictions = [task.result for task in party.gather(predict_tasks, recv_results=True)]

                predictions = self.aggregate(party.members, party_predictions, is_infer=True)
                target = self.target[[self.uid2tensor_idx[uid] for uid in batcher.uids]]
                self.report_metrics(target, predictions, name="Train", step=titer.seq_num)

            if self.report_test_metrics_iteration > 0 and titer.seq_num % self.report_test_metrics_iteration == 0:
                logger.debug(
                    f"Master {self.id}: train loop - reporting test metrics on iteration {titer.seq_num} "
                    f"of epoch {titer.epoch}"
                )
                predict_test_tasks = party.broadcast(
                    Method.predict,
                    method_kwargs=MethodKwargs(other_kwargs={"uids": self.inference_target_uids, "is_infer": True}),
                    participating_members=titer.participating_members,
                )

                party_predictions_test = [
                    task.result for task in party.gather(predict_test_tasks, recv_results=True)
                ]

                predictions = self.aggregate(party.members, party_predictions_test, is_infer=True)
                test_target = self.test_target[[self.uid2tensor_idx_test[uid] for uid in self.inference_target_uids]]

                self.report_metrics(test_target, predictions, name="Test", step=titer.seq_num)

            self.iteration_times.append(
                IterationTime(client_id=self.id, iteration=titer.seq_num, iteration_time=time.time() - iter_start_time)
            )

    def inference_loop(self, batcher: Batcher, party: PartyCommunicator) -> None:
        """ Run main inference loop on the VFL master.

        :param batcher: Batcher for creating validation batches.
        :param party: Communicator instance used for communication between VFL agents.

        :return: None
        """
        logger.info(f"Master {self.id}: entering inference loop")
        party_predictions_test = defaultdict(list)
        for titer in batcher:
            if titer.last_batch:
                break
            logger.info(
                f"Master {self.id}: inference loop - starting batch {titer.seq_num} (sub iter {titer.subiter_seq_num})"
            )
            predict_test_tasks = party.broadcast(
                Method.predict,
                method_kwargs=MethodKwargs(other_kwargs={"uids": titer.batch, "is_infer": True}),
                participating_members=titer.participating_members,
            )
            for task in party.gather(predict_test_tasks, recv_results=True):
                party_predictions_test[task.from_id].append(task.result)

        party_predictions_test = self._aggregate_batched_predictions(party.members, party_predictions_test)
        predictions = self.aggregate(party.members, party_predictions_test, is_infer=True)
        target = self.test_target[[self.uid2tensor_idx_test[uid] for uid in batcher.uids]]
        self.report_metrics(target, predictions, name="Test", step=-1)

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
    ovr = False

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
            model_params: dict = None,
            seed: int = None,
            device: str = 'cpu',
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
        self.seed = seed
        self.device = torch.device(device)

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
    def predict(self, uids: Optional[RecordsBatch], is_infer: bool = False) -> DataTensor:
        """ Make predictions using the initialized model.

        :param uids: Batch of record unique identifiers.
        :param is_infer: Flag indicating whether to use the test data.

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
        logger.info(f"Running member {self.id}")
        if self.do_train:
            self._run(party, is_infer=False)

        if self.do_predict:
            self._run(party, is_infer=True)

    def _run(self, party: PartyCommunicator, is_infer: bool = False):
        # sync train
        synchronize_uids_task = party.recv(
            Task(method_name=Method.records_uids, from_id=party.master, to_id=self.id)
        )
        uids = self.execute_received_task(synchronize_uids_task)
        party.send(party.master, Method.records_uids, result=uids)

        initialize_task = party.recv(Task(method_name=Method.initialize, from_id=party.master, to_id=self.id))
        self.execute_received_task(initialize_task)

        register_records_uids_task = party.recv(
            Task(method_name=Method.register_records_uids, from_id=party.master, to_id=self.id)
        )
        self.execute_received_task(register_records_uids_task)

        if not is_infer:
            # sync test
            synchronize_uids_task = party.recv(
                Task(method_name=Method.records_uids, from_id=party.master, to_id=self.id)
            )
            uids = self.execute_received_task(synchronize_uids_task)
            party.send(party.master, Method.records_uids, result=uids)
            register_records_uids_task = party.recv(
                Task(method_name=Method.register_records_uids, from_id=party.master, to_id=self.id)
            )
            self.execute_received_task(register_records_uids_task)

            self.loop(batcher=self.make_batcher(), party=party)
        else:
            self.inference_loop(batcher=self.make_batcher(is_infer=True), party=party)

        finalize_task = party.recv(Task(method_name=Method.finalize, from_id=party.master, to_id=self.id))
        self.execute_received_task(finalize_task)
        logger.info(f"Finished member {self.id}")

    def _predict_metrics_loop(self, party: PartyCommunicator):
        predict_task = party.recv(Task(method_name=Method.predict, from_id=party.master, to_id=self.id))
        predictions = self.execute_received_task(predict_task)
        party.send(party.master, Method.predict, result=predictions)

    def loop(self, batcher: Batcher, party: PartyCommunicator):
        """ Run main training loop on the VFL member.

        :param batcher: Batcher for creating training batches.
        :param party: Communicator instance used for communication between VFL agents.

        :return: None
        """
        logger.info(f"Member {self.id}: entering training loop")
        for titer in batcher:
            if titer.participating_members is not None:
                if self.id not in titer.participating_members:
                    logger.debug(f'Member {self.id} skipping {titer.seq_num}.')
                    continue
            logger.info(
                f"Member {self.id}: train loop - starting batch {titer.seq_num} (sub iter {titer.subiter_seq_num}) "
                f"on epoch {titer.epoch}"
            )
            update_predict_task = party.recv(
                Task(method_name=Method.update_predict, from_id=party.master, to_id=self.id)
            )
            predictions = self.execute_received_task(update_predict_task)
            party.send(party.master, Method.update_predict, result=predictions)

            if self.report_train_metrics_iteration > 0 and titer.seq_num % self.report_train_metrics_iteration == 0:
                logger.debug(
                    f"Member {self.id}: train loop - reporting train metrics on iteration {titer.seq_num} "
                    f"of epoch {titer.epoch}"
                )
                self._predict_metrics_loop(party)

            if self.report_test_metrics_iteration > 0 and titer.seq_num % self.report_test_metrics_iteration == 0:
                logger.debug(
                    f"Member {self.id}: test loop - reporting train metrics on iteration {titer.seq_num} "
                    f"of epoch {titer.epoch}"
                )
                self._predict_metrics_loop(party)

    def inference_loop(self, batcher: Batcher, party: PartyCommunicator):
        """ Run main inference loop on the VFL member.

        :param batcher: Batcher for creating training batches.
        :param party: Communicator instance used for communication between VFL agents.

        :return: None
        """
        logger.info(f"Member {self.id}: entering inference loop")

        for titer in batcher:
            if titer.last_batch:
                break
            logger.info(
                f"Member {self.id}: inference loop - starting batch {titer.seq_num} (sub iter {titer.subiter_seq_num})"
            )
            if titer.participating_members is not None:
                if self.id not in titer.participating_members:
                    logger.debug(f'Member {self.id} skipping {titer.seq_num}.')
                    continue

            predict_task = party.recv(Task(method_name=Method.predict, from_id=party.master, to_id=self.id))
            predictions = self.execute_received_task(predict_task)
            party.send(party.master, Method.predict, result=predictions)

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
        logger.info(f"Member {self.id} makes a batcher for {len(uids_to_use)} uids")
        return self._create_batcher(epochs=epochs, uids=uids_to_use, batch_size=batch_size)

    def _create_batcher(self, epochs: int, uids: List[str], batch_size: int) -> Batcher:
        """Create a make_batcher for training.

        :param epochs: Number of training epochs.
        :param uids: List of unique identifiers for dataset rows.
        :param batch_size: Size of the training batch.
        """
        logger.info(f"Member {self.id}: making a make_batcher for {len(uids)} uids")
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
        logger.info(f"Member {self.id}: reporting existing record uids")
        if is_infer:
            return self._infer_uids, self.use_inner_join
        return self._uids, self.use_inner_join

    def register_records_uids(self, uids: List[str], is_infer: bool = False) -> None:
        """ Register unique identifiers to be used.

        :param uids: List of unique identifiers.
        :return: None
        """
        logger.info(f"Member {self.id}: registering {len(uids)} uids to be used.")

        if is_infer:
            self._uids_to_use_test = uids
        else:
            self._uids_to_use = uids
        self.fillna(is_infer=is_infer)

    def fillna(self, is_infer: bool = False) -> None:
        """ Fills missing values for member's dataset"""
        uids_to_use = self._uids_to_use_test if is_infer else self._uids_to_use
        _uids = self._infer_uids if is_infer else self._uids
        _uid2tensor_idx = self.uid2tensor_idx_test if is_infer else self.uid2tensor_idx
        split = self._data_params.test_split if is_infer else self._data_params.train_split

        uids_to_fill = list(set(uids_to_use) - set(_uids))
        if len(uids_to_fill) == 0:
            return

        logger.info(f"Member {self.id} has {len(uids_to_fill)} missing values: using fillna...")
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
        self.prepare_device_data(is_infer=is_infer)
        if not is_infer:
            self.initialize_model()
            self.initialize_optimizer()

    def prepare_device_data(self, is_infer: bool = False):
        if not is_infer:
            self.device_dataset_train_split = self._dataset[self._data_params.train_split][
                self._data_params.features_key].to(self.device)
        else:
            self.device_dataset_test_split = self._dataset[self._data_params.test_split][
                self._data_params.features_key].to(self.device)

    def initialize(self, is_infer: bool = False):
        """ Initialize the party member. """

        logger.info(f"Member {self.id}: initializing")
        self._dataset = self.processor.fit_transform()
        self.uid2tensor_idx = {uid: i for i, uid in enumerate(self._uids)}
        self.uid2tensor_idx_test = {uid: i for i, uid in enumerate(self._infer_uids)}
        self._data_params = self.processor.data_params
        self._common_params = self.processor.common_params
        self.initialize_model(do_load_model=is_infer)
        self.initialize_optimizer()
        self.prepare_device_data(is_infer=True)
        self.prepare_device_data(is_infer=False)
        self.is_initialized = True
        self.is_finalized = False
        logger.info(f"Member {self.id}: has been initialized")

    def finalize(self, is_infer: bool = False) -> None:
        """ Finalize the party member. """
        logger.info(f"Member {self.id}: finalizing")
        self.check_if_ready()
        if self.do_save_model and not is_infer:
            self.save_model(is_ovr_models=self.ovr)
        self.is_finalized = True
        logger.info(f"Member {self.id}: has finalized")

    def _prepare_data(self, uids: RecordsBatch) -> Tuple:
        """ Prepare data for training.

        :param uids: Batch of record unique identifiers.
        :return: Tuple of three SVD matrices.
        """
        X_train = self.device_dataset_train_split[[int(x) for x in uids]]
        U, S, Vh = sp.linalg.svd(X_train.numpy(), full_matrices=False, overwrite_a=False, check_finite=False)
        return U, S, Vh
