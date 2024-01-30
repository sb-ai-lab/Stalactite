import collections
import enum
import itertools
import logging
import time
from abc import ABC, abstractmethod
from concurrent.futures import Future
from dataclasses import dataclass, field
from typing import Any, Iterator, List, Optional, Union

import torch

logger = logging.getLogger(__name__)

logger.setLevel(logging.DEBUG)
sh = logging.StreamHandler()
sh.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
logger.addHandler(sh)

DataTensor = torch.Tensor
# in reality, it will be a DataTensor but with one more dimension
PartyDataTensor = List[torch.Tensor]

RecordsBatch = List[str]


class Method(str, enum.Enum):  # TODO _Method the same - unify
    service_return_answer = "service_return_answer"
    service_heartbeat = "service_heartbeat"

    records_uids = "records_uids"
    register_records_uids = "register_records_uids"

    initialize = "initialize"
    finalize = "finalize"

    update_weights = "update_weights"
    predict = "predict"
    update_predict = "update_predict"


@dataclass(frozen=True)
class TrainingIteration:
    seq_num: int
    subiter_seq_num: int
    epoch: int
    batch: RecordsBatch
    previous_batch: Optional[RecordsBatch]
    participating_members: Optional[List[str]]


@dataclass
class MethodKwargs:  # TODO MethodMessage the same - unify
    """Data class holding keyword arguments for called method."""

    tensor_kwargs: dict[str, torch.Tensor] = field(default_factory=dict)
    other_kwargs: dict[str, Any] = field(default_factory=dict)


class Batcher(ABC):
    # todo: add docs
    uids: List[str]

    @abstractmethod
    def __iter__(self) -> Iterator[TrainingIteration]:
        ...


class ParticipantFuture(Future):
    def __init__(self, participant_id: str):
        super().__init__()
        self.participant_id = participant_id


@dataclass
class Task:
    method_name: str
    from_id: str
    to_id: str
    id: Optional[str] = None
    method_kwargs: Optional[MethodKwargs] = None
    result: Any = None

    @property
    def kwargs_dict(self) -> dict:
        if self.method_kwargs is not None:
            return {**self.method_kwargs.other_kwargs, **self.method_kwargs.tensor_kwargs}
        else:
            return dict()


class PartyCommunicator(ABC):
    # todo: add docs
    world_size: int
    # todo: add docs about order guaranteeing
    members: List[str]

    @abstractmethod
    def rendezvous(self):
        ...

    @property
    @abstractmethod
    def is_ready(self) -> bool:
        ...

    @abstractmethod
    def send(
        self,
        send_to_id: str,
        method_name: Method,
        method_kwargs: Optional[MethodKwargs] = None,
        result: Optional[Any] = None,
        **kwargs,
    ) -> Task:
        ...

    @abstractmethod
    def recv(self, task: Task, recv_results: bool = False) -> Task:  # TODO
        ...

    @abstractmethod
    def broadcast(
        self,
        method_name: Method,
        method_kwargs: Optional[MethodKwargs] = None,
        result: Optional[Any] = None,
        participating_members: Optional[List[str]] = None,
        include_current_participant: bool = False,
        **kwargs,
    ) -> List[Task]:
        ...

    @abstractmethod
    def scatter(
        self,
        method_name: Method,
        method_kwargs: Optional[List[MethodKwargs]] = None,
        result: Optional[Any] = None,
        participating_members: Optional[List[str]] = None,
        **kwargs,
    ) -> List[Task]:  # TODO
        ...

    @abstractmethod
    def gather(self, tasks: List[Task], recv_results: bool = False) -> List[Task]:  # TODO
        ...

    @abstractmethod
    def run(self):
        ...


class PartyMaster(ABC):
    # todo: add docs
    id: str
    epochs: int
    report_train_metrics_iteration: int
    report_test_metrics_iteration: int
    target: DataTensor
    target_uids: List[str]
    test_target: DataTensor

    _iter_time: list[tuple[int, float]] = list()

    @property
    def train_timings(self) -> list:
        return self._iter_time

    def run(self, communicator: PartyCommunicator):  # TODO rename to party
        logger.info("Running master %s" % self.id)

        records_uids_tasks = communicator.broadcast(
            Method.records_uids,
            participating_members=communicator.members,
        )

        records_uids_results = communicator.gather(records_uids_tasks, recv_results=True)

        collected_uids_results = [task.result for task in records_uids_results]

        uids = self.synchronize_uids(collected_uids_results, world_size=communicator.world_size)
        communicator.broadcast(
            Method.register_records_uids,
            method_kwargs=MethodKwargs(other_kwargs={"uids": uids}),
            participating_members=communicator.members,
        )

        communicator.broadcast(
            Method.initialize,
            participating_members=communicator.members,
        )
        self.initialize()

        self.loop(batcher=self.make_batcher(uids=uids, party_members=communicator.members), communicator=communicator)

        communicator.broadcast(
            Method.finalize,
            participating_members=communicator.members,
        )
        self.finalize()
        logger.info("Finished master %s" % self.id)

    def loop(self, batcher: Batcher, communicator: PartyCommunicator):
        logger.info("Master %s: entering training loop" % self.id)
        updates = self.make_init_updates(communicator.world_size)
        for titer in batcher:
            logger.debug(
                f"Master %s: train loop - starting batch %s (sub iter %s) on epoch %s"
                % (self.id, titer.seq_num, titer.subiter_seq_num, titer.epoch)
            )
            iter_start_time = time.time()
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

            party_predictions = [task.result for task in communicator.gather(update_predict_tasks, recv_results=True)]

            predictions = self.aggregate(titer.participating_members, party_predictions)

            updates = self.compute_updates(
                titer.participating_members,
                predictions,
                party_predictions,
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
                party_predictions = [task.result for task in communicator.gather(predict_tasks, recv_results=True)]

                predictions = self.aggregate(communicator.members, party_predictions, infer=True)
                self.report_metrics(self.target, predictions, name="Train")

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

                party_predictions_test = [
                    task.result for task in communicator.gather(predict_test_tasks, recv_results=True)
                ]

                predictions = self.aggregate(communicator.members, party_predictions_test, infer=True)
                self.report_metrics(self.test_target, predictions, name="Test")

            self._iter_time.append((titer.seq_num, time.time() - iter_start_time))

    def synchronize_uids(self, collected_uids: list[list[str]], world_size: int) -> List[str]:
        logger.debug("Master %s: synchronizing uids for party of size %s" % (self.id, world_size))
        uids = itertools.chain(self.target_uids, (uid for member_uids in collected_uids for uid in set(member_uids)))
        shared_uids = sorted([uid for uid, count in collections.Counter(uids).items() if count == world_size + 1])
        logger.debug("Master %s: registering shared uids f size %s" % (self.id, len(shared_uids)))
        set_shared_uids = set(shared_uids)
        uid2idx = {uid: i for i, uid in enumerate(self.target_uids) if uid in set_shared_uids}
        selected_tensor_idx = [uid2idx[uid] for uid in shared_uids]

        self.target = self.target[selected_tensor_idx]
        self.target_uids = shared_uids
        logger.debug("Master %s: record uids has been successfully synchronized")
        return shared_uids

    @abstractmethod
    def make_batcher(self, uids: List[str], party_members: List[str]) -> Batcher:
        ...

    @abstractmethod
    def initialize(self):
        ...

    @abstractmethod
    def finalize(self):
        ...

    @abstractmethod
    def make_init_updates(self, world_size: int) -> PartyDataTensor:
        ...

    @abstractmethod
    def aggregate(
        self, participating_members: List[str], party_predictions: PartyDataTensor, infer: bool = False
    ) -> DataTensor:
        ...

    @abstractmethod
    def compute_updates(
        self,
        participating_members: List[str],
        predictions: DataTensor,
        party_predictions: PartyDataTensor,
        world_size: int,
        subiter_seq_num: int,
    ) -> List[DataTensor]:
        ...

    @abstractmethod
    def report_metrics(self, y: DataTensor, predictions: DataTensor, name: str):
        ...


class PartyMember(ABC):
    # todo: add docs
    id: str
    master_id: str
    report_train_metrics_iteration: int
    report_test_metrics_iteration: int
    _iter_time: list[tuple[int, float]] = list()

    def _execute_received_task(self, task: Task) -> Optional[Union[DataTensor, List[str]]]:
        return getattr(self, task.method_name)(**task.kwargs_dict)

    def run(self, communicator: PartyCommunicator):  # TODO
        logger.info("Running member %s" % self.id)

        synchronize_uids_task = communicator.recv(
            Task(method_name=Method.records_uids, from_id=self.master_id, to_id=self.id)
        )
        uids = self._execute_received_task(synchronize_uids_task)
        communicator.send(self.master_id, Method.records_uids, result=uids)

        register_records_uids_task = communicator.recv(
            Task(method_name=Method.register_records_uids, from_id=self.master_id, to_id=self.id)
        )
        self._execute_received_task(register_records_uids_task)

        initialize_task = communicator.recv(Task(method_name=Method.initialize, from_id=self.master_id, to_id=self.id))
        self._execute_received_task(initialize_task)

        self.loop(batcher=self.batcher, communicator=communicator)

        finalize_task = communicator.recv(Task(method_name=Method.finalize, from_id=self.master_id, to_id=self.id))
        self._execute_received_task(finalize_task)
        logger.info("Finished member %s" % self.id)

    def loop(self, batcher: Batcher, communicator: PartyCommunicator):
        logger.info("Member %s: entering training loop" % self.id)

        for titer in batcher:
            logger.debug(
                f"Member %s: train loop - starting batch %s (sub iter %s) on epoch %s"
                % (self.id, titer.seq_num, titer.subiter_seq_num, titer.epoch)
            )
            iter_start_time = time.time()
            update_predict_task = communicator.recv(
                Task(method_name=Method.update_predict, from_id=self.master_id, to_id=self.id)
            )
            predictions = self._execute_received_task(update_predict_task)
            communicator.send(
                self.master_id,
                Method.update_predict,
                result=predictions,
            )

            if self.report_train_metrics_iteration > 0 and titer.seq_num % self.report_train_metrics_iteration == 0:
                logger.debug(
                    f"Member %s: train loop - calculating train metrics on iteration %s of epoch %s"
                    % (self.id, titer.seq_num, titer.epoch)
                )
                predict_task = communicator.recv(
                    Task(method_name=Method.predict, from_id=self.master_id, to_id=self.id)
                )
                predictions = self._execute_received_task(predict_task)
                communicator.send(self.master_id, Method.predict, result=predictions)

            if self.report_test_metrics_iteration > 0 and titer.seq_num % self.report_test_metrics_iteration == 0:
                logger.debug(
                    f"Member %s: train loop - calculating train metrics on iteration %s of epoch %s"
                    % (self.id, titer.seq_num, titer.epoch)
                )
                predict_task = communicator.recv(
                    Task(method_name=Method.predict, from_id=self.master_id, to_id=self.id)
                )
                predictions = self._execute_received_task(predict_task)
                communicator.send(self.master_id, Method.predict, result=predictions)

            self._iter_time.append((titer.seq_num, time.time() - iter_start_time))

    @property
    @abstractmethod
    def batcher(self) -> Batcher:
        ...

    @abstractmethod
    def records_uids(self) -> List[str]:
        ...

    @abstractmethod
    def register_records_uids(self, uids: List[str]):
        ...

    @abstractmethod
    def initialize(self):
        ...

    @abstractmethod
    def finalize(self):
        ...

    @abstractmethod
    def update_weights(self, uids: RecordsBatch, upd: DataTensor):
        ...

    @abstractmethod
    def predict(self, uids: RecordsBatch, use_test: bool = False) -> DataTensor:
        ...

    @abstractmethod
    def update_predict(self, upd: DataTensor, previous_batch: RecordsBatch, batch: RecordsBatch) -> DataTensor:
        ...
