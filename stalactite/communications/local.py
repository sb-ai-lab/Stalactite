import concurrent.futures
import logging
import threading
import time
import uuid
from abc import ABC
from collections import defaultdict
from concurrent.futures import Future
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

from stalactite.base import (
    Method,
    MethodKwargs,
    PartyCommunicator,
    PartyMaster,
    PartyMember,
    Task,
)
from stalactite.communications.helpers import ParticipantType

logger = logging.getLogger(__name__)


class RecvFuture(Future):
    def __init__(self, method_name: str, receive_from_id: str):
        super().__init__()
        self.method_name = method_name
        self.receive_from_id = receive_from_id


@dataclass(frozen=True)
class _ParticipantInfo:
    type: ParticipantType
    received_tasks: dict


class LocalPartyCommunicator(PartyCommunicator, ABC):
    """Base local communicator class."""

    MEMBER_DATA_FIELDNAME = "__member_data__"

    # todo: add docs
    # todo: introduce the single interface for that
    participant: Union[PartyMaster, PartyMember]
    _party_info: Optional[Dict[str, _ParticipantInfo]]
    recv_timeout: float
    master_id: Optional[str]
    _lock = threading.Lock()

    @property
    def is_ready(self) -> bool:
        """Whether the communicator is ready."""
        return self._party_info is not None

    def rendezvous(self):
        """Wait until all the participants of the party are initialized."""
        logger.info("Party communicator %s: performing rendezvous" % self.participant.id)
        self._party_info[self.participant.id] = _ParticipantInfo(
            type=ParticipantType.master if isinstance(self.participant, PartyMaster) else ParticipantType.member,
            received_tasks=defaultdict(dict),
        )

        # todo: allow to work with timeout for rendezvous operation
        while len(self._party_info) < self.world_size + 1:
            time.sleep(0.1)

        self.members = [uid for uid, pinfo in self._party_info.items() if pinfo.type == ParticipantType.member]

        logger.info("Party communicator %s: rendezvous has been successfully performed" % self.participant.id)

    def send(
            self,
            send_to_id: str,
            method_name: Method,
            method_kwargs: Optional[MethodKwargs] = None,
            result: Optional[Any] = None,
            **kwargs,
    ) -> Task:
        self._check_if_ready()
        if send_to_id not in self._party_info:
            raise ValueError(f"Unknown receiver: {send_to_id}")
        task = Task(
            method_name=method_name,
            from_id=self.participant.id,
            to_id=send_to_id,
            id=str(uuid.uuid4()),
            method_kwargs=method_kwargs,
            result=result,
        )
        with self._lock:
            self._party_info[send_to_id].received_tasks[task.method_name][self.participant.id] = task
        return task

    def _check_recv_tasks(self, method_name: str, receive_from_id: str) -> Task:
        timer_start = time.time()
        task = None
        while task is None:
            with self._lock:
                task = self._party_info[self.participant.id] \
                    .received_tasks \
                    .get(method_name, dict()) \
                    .pop(receive_from_id, None)
            if time.time() - timer_start > self.recv_timeout:
                raise TimeoutError(f"Could not receive task: {method_name} from {receive_from_id}.")
        return task

    def recv(self, task: Task, recv_results: bool = False) -> Task:
        return self._check_recv_tasks(
            method_name=task.method_name,
            receive_from_id=task.to_id if recv_results else task.from_id,
        )

    def _get_from_recv(self, recv_future: RecvFuture) -> None:
        result = self._check_recv_tasks(
            method_name=recv_future.method_name,
            receive_from_id=recv_future.receive_from_id,
        )
        recv_future.set_result(result)
        recv_future.done()

    def scatter(
            self,
            method_name: Method,
            method_kwargs: Optional[List[MethodKwargs]] = None,
            result: Optional[Any] = None,
            participating_members: Optional[List[str]] = None,
            **kwargs,
    ) -> List[Task]:

        if participating_members is None:
            participating_members = self.members

        assert (
                len(set(participating_members).difference(set(self.members))) == 0
        ), "Some of the members are disconnected, cannot perform collective operation"

        if method_kwargs is not None:
            assert len(method_kwargs) == len(participating_members), (
                f"Number of tasks in scatter operation ({len(method_kwargs)}) is not equal to the "
                f"`participating_members` number ({len(participating_members)})"
            )
        else:
            method_kwargs = [None for _ in range(len(participating_members))]

        tasks = []
        for send_to_id, m_kwargs in zip(participating_members, method_kwargs):
            tasks.append(
                self.send(
                    send_to_id=send_to_id,
                    method_name=method_name,
                    method_kwargs=m_kwargs,
                    result=result,
                    **kwargs,
                )
            )
        return tasks

    def broadcast(
            self,
            method_name: Method,
            method_kwargs: Optional[MethodKwargs] = None,
            result: Optional[Any] = None,
            participating_members: Optional[List[str]] = None,
            include_current_participant: bool = False,
            **kwargs,
    ) -> List[Task]:
        if self.participant.id not in participating_members and include_current_participant:
            participating_members.append(self.participant.id)

        if method_kwargs is not None and not isinstance(method_kwargs, MethodKwargs):
            raise TypeError(
                "communicator.broadcast `method_kwargs` must be either None or MethodKwargs. "
                f"Got {type(method_kwargs)}"
            )

        tasks = []
        for send_to_id in participating_members:
            tasks.append(
                self.send(
                    send_to_id=send_to_id,
                    method_name=method_name,
                    method_kwargs=method_kwargs,
                    result=result,
                    **kwargs,
                )
            )
        return tasks

    def gather(self, tasks: List[Task], recv_results: bool = False) -> List[Task]:
        _recv_futures = [
            RecvFuture(method_name=task.method_name, receive_from_id=task.from_id if not recv_results else task.to_id)
            for task in tasks
        ]
        for recv_f in _recv_futures:
            threading.Thread(target=self._get_from_recv, args=(recv_f,), daemon=True).start()
        done_tasks, pending_tasks = concurrent.futures.wait(_recv_futures, timeout=self.recv_timeout)
        if number_to := len(pending_tasks):
            raise TimeoutError(f"{self.participant.id} could not gather tasks from {number_to} members.")
        return [task.result() for task in done_tasks]

    def _check_if_ready(self):
        """Raise an exception if the communicator was not initialized properly."""
        if not self.is_ready:
            raise RuntimeError(
                "Cannot proceed because communicator is not ready. "
                "Perhaps, rendezvous has not been called or was unsuccessful"
            )


class LocalMasterPartyCommunicator(LocalPartyCommunicator):
    """Local Master communicator class.
    This class is used as the communicator for master in local (single-process) VFL setup."""

    def __init__(
            self,
            participant: PartyMaster,
            world_size: int,
            shared_party_info: Optional[Dict[str, _ParticipantInfo]],
            recv_timeout: float = 360.0,
    ):
        """
        Initialize master communicator with parameters.
        :param participant: PartyMaster instance
        :param world_size: Number of VFL member agents
        :param shared_party_info: Dictionary with agents _ParticipantInfo
        """
        self.participant = participant
        self.world_size = world_size
        self.recv_timeout = recv_timeout
        self.master_id = self.participant.id

        self._party_info: Optional[Dict[str, _ParticipantInfo]] = shared_party_info
        self._event_futures: Optional[Dict[str, Future]] = dict()

    def run(self):
        """
        Run the VFL master.
        Wait until rendezvous is finished and start sending, receiving and processing tasks from the main loop.
        """
        try:
            logger.info("Party communicator %s: running" % self.participant.id)
            self.rendezvous()

            self.participant.run(communicator=self)

            logger.info("Party communicator %s: finished" % self.participant.id)
        except:
            logger.error("Exception in party communicator %s" % self.participant.id, exc_info=True)
            raise


class LocalMemberPartyCommunicator(LocalPartyCommunicator):
    """gRPC Member communicator class.
    This class is used as the communicator for member in local (single-process) VFL setup."""

    def __init__(
            self,
            participant: PartyMember,
            master_id: str,
            world_size: int,
            shared_party_info: Optional[Dict[str, _ParticipantInfo]],
            recv_timeout: float = 360.0,
    ):
        """
        Initialize member communicator with parameters.
        :param participant: PartyMember instance
        :param world_size: Number of VFL member agents
        :param shared_party_info: Dictionary with agents _ParticipantInfo
        """
        self.participant = participant
        self.world_size = world_size
        self.recv_timeout = recv_timeout
        self.master_id = master_id
        self._party_info: Optional[Dict[str, _ParticipantInfo]] = shared_party_info
        self._event_futures: Optional[Dict[str, Future]] = dict()

    def run(self):
        """
        Run the VFL member.
        Perform rendezvous. Start main events processing loop.
        """
        try:
            logger.info("Party communicator %s: running" % self.participant.id)
            self.rendezvous()
            self.participant.master_id = self.master_id
            self.participant.run(communicator=self)
            logger.info("Party communicator %s: finished" % self.participant.id)
        except:
            logger.error("Exception in party communicator %s" % self.participant.id, exc_info=True)
            raise
