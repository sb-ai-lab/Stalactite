import concurrent.futures
import enum
import logging
import time
import uuid
from abc import ABC
from concurrent.futures import Future
from copy import copy
from dataclasses import dataclass
from queue import Queue
from threading import Thread
from typing import List, Dict, Any, Optional, Union, cast, Set

from stalactite.base import PartyMaster, PartyMember, PartyCommunicator, ParticipantFuture, Party, PartyDataTensor, \
    RecordsBatch
from stalactite.communications.helpers import ParticipantType, _Method

logger = logging.getLogger(__name__)


@dataclass
class _Event:
    id: str
    parent_id: Optional[str]
    from_uid: str
    method_name: str
    data: Dict[str, Any]

    def __repr__(self) -> str:
        return f"_Event(id={self.id}, method_name={self.method_name}, " \
               f"from_id={self.from_uid}, parent_id={self.parent_id})"


@dataclass(frozen=True)
class _ParticipantInfo:
    type: ParticipantType
    queue: Queue


class LocalPartyCommunicator(PartyCommunicator, ABC):
    """ Base local communicator class. """
    MEMBER_DATA_FIELDNAME = '__member_data__'

    # todo: add docs
    # todo: introduce the single interface for that
    participant: Union[PartyMaster, PartyMember]
    _party_info: Optional[Dict[str, _ParticipantInfo]]
    _event_futures: Optional[Dict[str, ParticipantFuture]]

    @property
    def is_ready(self) -> bool:
        """ Whether the communicator is ready. """
        return self._party_info is not None

    def randezvous(self):
        """ Wait until all the participants of the party are initialized. """
        logger.info("Party communicator %s: performing randezvous" % self.participant.id)
        self._party_info[self.participant.id] = _ParticipantInfo(
            type=ParticipantType.master if isinstance(self.participant, PartyMaster) else ParticipantType.member,
            queue=Queue()
        )

        # todo: allow to work with timeout for randezvous operation
        while len(self._party_info) < self.world_size + 1:
            time.sleep(0.1)

        self.members = [uid for uid, pinfo in self._party_info.items() if pinfo.type == ParticipantType.member]

        logger.info("Party communicator %s: randezvous has been successfully performed" % self.participant.id)

    def send(self, send_to_id: str, method_name: str, parent_id: Optional[str] = None,
             require_answer: bool = True, **kwargs) -> ParticipantFuture:
        """
        Send task to VFL member or master.

        :param send_to_id: Identifier of the VFL agent receiver
        :param method_name: Method name to execute on participant
        :param require_answer: True if the task requires answer to sent back to sender
        :param parent_id: Unique identifier of the parent task
        :param kwargs: Keyword arguments to send
        """
        self._check_if_ready()

        if send_to_id not in self._party_info:
            raise ValueError(f"Unknown receiver: {send_to_id}")

        event = _Event(id=str(uuid.uuid4()), parent_id=parent_id, from_uid=self.participant.id,
                       method_name=method_name, data=kwargs)

        return self._publish_message(event, send_to_id, require_answer=require_answer)

    def broadcast(self,
                  method_name: str,
                  mass_kwargs: Optional[List[Any]] = None,
                  participating_members: Optional[List[str]] = None,
                  parent_id: Optional[str] = None,
                  require_answer: bool = True,
                  include_current_participant: bool = False,
                  **kwargs) -> List[ParticipantFuture]:
        """
        Broadcast tasks to VFL agents.

        :param method_name: Method name to execute on participant
        :param mass_kwargs: List of the torch.Tensors to send to each member in `participating_members`, respectively
        :param participating_members: List of the members to which the task will be broadcasted
        :param parent_id: Unique identifier of the parent task
        :param require_answer: True if the task requires answer to sent back to sender
        :param include_current_participant: True if the task should be performed by the VFL master too
        :param kwargs: Keyword arguments to send
        """
        self._check_if_ready()
        logger.debug("Sending event (%s) for all members" % method_name)

        members = participating_members or self.members

        unknown_members = set(members).difference(self.members)
        if len(unknown_members) > 0:
            raise ValueError(f"Unknown members: {unknown_members}. Existing members: {self.members}")

        if not mass_kwargs:
            mass_kwargs = [dict() for _ in members]
        elif mass_kwargs and len(mass_kwargs) != len(members):
            raise ValueError(f"Length of arguments list ({len(mass_kwargs)}) is not equal "
                             f"to the length of members ({len(members)})")
        else:
            mass_kwargs = [{self.MEMBER_DATA_FIELDNAME: args} for args in mass_kwargs]

        futures = []

        for mkwargs, member_id in zip(mass_kwargs, members):
            if member_id == self.participant.id and not include_current_participant:
                continue

            event = _Event(id=str(uuid.uuid4()), parent_id=parent_id, from_uid=self.participant.id,
                           method_name=method_name, data={**mkwargs, **kwargs})

            future = self._publish_message(event, member_id, require_answer=require_answer)
            if future:
                futures.append(future)

        return futures

    def _publish_message(self, event: _Event, receiver_id: str, require_answer: bool) -> Optional[ParticipantFuture]:
        """ Create a future for the Event. """
        logger.debug("Party communicator %s: sending to %s event %s" % (self.participant.id, receiver_id, event))

        future = ParticipantFuture(participant_id=receiver_id)
        self._party_info[receiver_id].queue.put(event)

        # not all command requires feedback
        if require_answer:
            self._event_futures[event.id] = future
        else:
            future.set_result(None)

        logger.debug("Party communicator %s: sent to %s event %s" % (self.participant.id, receiver_id, event.id))

        return future

    def _check_if_ready(self):
        """ Raise an exception if the communicator was not initialized properly. """
        if not self.is_ready:
            raise RuntimeError("Cannot proceed because communicator is not ready. "
                               "Perhaps, randezvous has not been called or was unsuccessful")


class LocalMasterPartyCommunicator(LocalPartyCommunicator):
    """ Local Master communicator class.
    This class is used as the communicator for master in local (single-process) VFL setup.
    """

    def __init__(
            self,
            participant: PartyMaster,
            world_size: int,
            shared_party_info: Optional[Dict[str, _ParticipantInfo]]
    ):
        """
        Initialize master communicator with parameters.

        :param participant: PartyMaster instance
        :param world_size: Number of VFL member agents
        :param shared_party_info: Dictionary with agents _ParticipantInfo
        """
        self.participant = participant
        self.world_size = world_size

        self._party_info: Optional[Dict[str, _ParticipantInfo]] = shared_party_info
        self._event_futures: Optional[Dict[str, Future]] = dict()

    def run(self):
        """
        Run the VFL master.
        Wait until rendezvous is finished and start sending, receiving and processing tasks from the main loop.
        """
        try:
            logger.info("Party communicator %s: running" % self.participant.id)
            self.randezvous()
            party = LocalPartyImpl(party_communicator=self)

            event_loop = Thread(name=f"event-loop-{self.participant.id}", daemon=True, target=self._run)
            event_loop.start()

            self.participant.run(party)

            self.send(send_to_id=self.participant.id, method_name=_Method.finalize.value, require_answer=False)
            event_loop.join()
            logger.info("Party communicator %s: finished" % self.participant.id)
        except:
            logger.error("Exception in party communicator %s" % self.participant.id, exc_info=True)
            raise

    def _run(self):
        """ Process tasks scheduled in the queue for VFL master. """
        try:
            logger.info("Party communicator %s: starting event loop" % self.participant.id)

            while True:
                event = self._party_info[self.participant.id].queue.get()

                logger.debug("Party communicator %s: received event %s" % (self.participant.id, event))

                if event.method_name == _Method.service_return_answer.value:
                    if event.parent_id not in self._event_futures:
                        # todo: replace with custom error
                        raise ValueError(f"No awaiting future with id {event.parent_id}."
                                         f"(Participant id {self.participant.id}. "
                                         f"Event {event.id} from {event.from_uid})")

                    logger.debug("Party communicator %s: marking future %s as finished by answer of event %s"
                                 % (self.participant.id, event.parent_id, event.id))

                    if 'result' not in event.data:
                        # todo: replace with custom error
                        raise ValueError("No result in data")

                    future = self._event_futures.pop(event.parent_id)
                    future.set_result(event.data['result'])
                    future.done()
                elif event.method_name == _Method.service_heartbeat.value:
                    logger.info("Party communicator %s: received heartbeat from %s: %s"
                                % (self.participant.id, event.id, event.data))
                elif event.method_name == _Method.finalize.value:
                    logger.info("Party communicator %s: finalized" % self.participant.id)
                    break
                else:
                    raise ValueError(f"Unsupported method {event.method_name} (Event {event.id} from {event.from_uid})")

            logger.info("Party communicator %s: finished event loop" % self.participant.id)
        except:
            logger.error("Exception in party communicator %s" % self.participant.id, exc_info=True)
            raise


class LocalMemberPartyCommunicator(LocalPartyCommunicator):
    """ gRPC Member communicator class.
    This class is used as the communicator for member in local (single-process) VFL setup.
    """

    def __init__(
            self,
            participant: PartyMember,
            world_size: int,
            shared_party_info: Optional[Dict[str, _ParticipantInfo]]
    ):
        """
        Initialize member communicator with parameters.

        :param participant: PartyMember instance
        :param world_size: Number of VFL member agents
        :param shared_party_info: Dictionary with agents _ParticipantInfo
        """
        self.participant = participant
        self.world_size = world_size
        self._party_info: Optional[Dict[str, _ParticipantInfo]] = shared_party_info
        self._event_futures: Optional[Dict[str, Future]] = dict()

    def run(self):
        """
        Run the VFL member.
        Perform rendezvous. Start main events processing loop.
        """
        try:
            logger.info("Party communicator %s: running" % self.participant.id)
            self.randezvous()
            self._run()
            logger.info("Party communicator %s: finished" % self.participant.id)
        except:
            logger.error("Exception in party communicator %s" % self.participant.id, exc_info=True)
            raise

    def _run(self):
        """ Process tasks scheduled in the queue for VFL member. """
        logger.info("Party communicator %s: starting event loop" % self.participant.id)

        supported_methods = [
            _Method.records_uids.value,
            _Method.register_records_uids.value,
            _Method.initialize.value,
            _Method.finalize.value,
            _Method.update_weights.value,
            _Method.predict.value,
            _Method.update_predict.value
        ]

        not_found_methods = [
            method_name for method_name in supported_methods
            if hasattr(self.participant.id, method_name)
        ]
        if len(not_found_methods) > 0:
            raise RuntimeError(f"Found methods not supported by member {self.participant.id}: {not_found_methods}")

        while True:
            event = self._party_info[self.participant.id].queue.get()

            logger.debug("Party communicator %s: received event %s" % (self.participant.id, event))

            if event.method_name in supported_methods:
                method = getattr(self.participant, event.method_name)

                kwargs = copy(event.data)
                if self.MEMBER_DATA_FIELDNAME in kwargs:
                    mkwargs = [kwargs[self.MEMBER_DATA_FIELDNAME]]
                    del kwargs[self.MEMBER_DATA_FIELDNAME]
                else:
                    mkwargs = []

                result = method(*mkwargs, **kwargs)

                self.send(send_to_id=event.from_uid, method_name=_Method.service_return_answer.value,
                          parent_id=event.id, require_answer=False, result=result)

                if event.method_name == _Method.finalize.value:
                    break
            else:
                raise ValueError(f"Unsupported method {event.method_name} (Event {event.id} from {event.from_uid})")

        logger.info("Party communicator %s: finished event loop" % self.participant.id)


class LocalPartyImpl(Party):
    """ Helper Party class for tasks scheduling. """

    # todo: add logging to the methods
    def __init__(self, party_communicator: PartyCommunicator, op_timeout: Optional[float] = None):
        """
        Initialize LocalPartyImpl with the communicator.

        :param party_communicator: VFL master communicator instance
        :param op_timeout: Maximum time to perform tasks in seconds
        """
        self.party_communicator = party_communicator
        self.op_timeout = op_timeout

    @property
    def world_size(self) -> int:
        """ Return number of the VFL members. """
        return self.party_communicator.world_size

    @property
    def members(self) -> List[str]:
        """ Return list of the VFL members. """
        return self.party_communicator.members

    def _sync_broadcast_to_members(
            self, *,
            method_name: _Method,
            mass_kwargs: Optional[List[Any]] = None,
            participating_members: Optional[List[str]] = None,
            **kwargs
    ) -> List[Any]:
        """
        Broadcast tasks to VFL agents using master communicator.

        :param method_name: Method name to execute on participant
        :param mass_kwargs: List of the torch.Tensors to send to each member in `participating_members`, respectively
        :param participating_members: List of the members to which the task will be broadcasted
        :param kwargs: Keyword arguments to send
        """
        futures = self.party_communicator.broadcast(
            method_name=method_name.value,
            participating_members=participating_members,
            mass_kwargs=mass_kwargs,
            **kwargs
        )

        logger.debug("Party broadcast: waiting for answer to event %s (waiting for %s secs)"
                     % (method_name, self.op_timeout or "inf"))

        futures = concurrent.futures.wait(futures, timeout=self.op_timeout)
        completed_futures, uncompleted_futures \
            = cast(Set[ParticipantFuture], futures[0]), cast(Set[ParticipantFuture], futures[1])

        if len(uncompleted_futures) > 0:
            # todo: custom exception with additional info about uncompleted tasks
            raise RuntimeError(f"Not all tasks have been completed. "
                               f"Completed tasks: {len(completed_futures)}. "
                               f"Uncompleted tasks: {len(uncompleted_futures)}.")

        logger.debug("Party broadcast for event %s has succesfully finished" % method_name)

        fresults = {future.participant_id: future.result() for future in completed_futures}
        return [fresults[member_id] for member_id in self.party_communicator.members]

    def records_uids(self) -> List[List[str]]:
        """ Collect records uids from VFL members. """
        return cast(List[List[str]], self._sync_broadcast_to_members(method_name=_Method.records_uids))

    def register_records_uids(self, uids: List[str]):
        """ Register records uids in VFL agents. """
        self._sync_broadcast_to_members(method_name=_Method.register_records_uids, uids=uids)

    def initialize(self):
        """ Initialize party communicators. """
        self._sync_broadcast_to_members(method_name=_Method.initialize)

    def finalize(self):
        """ Finalize party communicators. """
        self._sync_broadcast_to_members(method_name=_Method.finalize)

    def update_weights(self, upd: PartyDataTensor):
        """ Update model`s weights on agents. """
        self._sync_broadcast_to_members(method_name=_Method.update_weights, mass_kwargs=upd)

    def predict(self, uids: List[str], use_test: bool = False) -> PartyDataTensor:
        """ Make predictions using VFL agents` models. """
        return cast(
            PartyDataTensor,
            self._sync_broadcast_to_members(method_name=_Method.predict, uids=uids, use_test=True)
        )

    def update_predict(
            self,
            participating_members: List[str],
            batch: RecordsBatch,
            previous_batch: RecordsBatch,
            upd: PartyDataTensor
    ) -> PartyDataTensor:
        """ Perform update_predict operation on the VFL agents. """
        return cast(
            PartyDataTensor,
            self._sync_broadcast_to_members(
                method_name=_Method.update_predict,
                mass_kwargs=upd,
                participating_members=participating_members,
                batch=batch,
                previous_batch=previous_batch
            )
        )
