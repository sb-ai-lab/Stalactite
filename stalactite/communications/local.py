import concurrent.futures
import enum
import logging
import time
import uuid
from abc import abstractmethod
from concurrent.futures import Future
from dataclasses import dataclass
from queue import Queue
from threading import Thread
from typing import List, Dict, Any, Optional, Union, cast, Set

from stalactite.base import PartyMaster, PartyMember, PartyCommunicator, ParticipantFuture, Party, PartyDataTensor

logger = logging.getLogger(__name__)


class _Method(enum.Enum):
    service_return_answer = 'service_return_answer'
    service_heartbeat = 'service_heartbeat'

    record_uids = 'synchronize_uids'
    register_records_uids = 'register_records_uids'

    initialize = 'initialize'
    finalize = 'finalize'

    update_weights = 'update_weights'
    predict = 'predict'
    update_predict = 'update_predict'


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
    queue: Queue


class LocalPartyCommunicator(PartyCommunicator):
    # todo: add docs
    # todo: introduce the single interface for that
    participant: Union[PartyMaster, PartyMember]
    _party_info: Optional[Dict[str, _ParticipantInfo]]
    _event_futures: Optional[Dict[str, ParticipantFuture]]

    @property
    def is_ready(self) -> bool:
        return self._party_info is not None

    def randezvous(self):
        logger.info("Party communicator %s: performing randezvous" % self.participant.id)
        self._party_info[self.participant.id] = _ParticipantInfo(queue=Queue())

        # todo: allow to work with timeout for randezvous operation
        while len(self._party_info) < self.world_size:
            time.sleep(0.1)

        logger.info("Party communicator %s: randezvous has been successfully performed" % self.participant.id)

    def run(self):
        logger.info("Party communicator %s: running" % self.participant.id)
        self.randezvous()
        self._run()
        logger.info("Party communicator %s: finished" % self.participant.id)

    def send(self, send_to_id: str, method_name: str, parent_id: Optional[str] = None, **kwargs) -> ParticipantFuture:
        self._check_if_ready()

        if send_to_id not in self._party_info:
            raise ValueError(f"Unknown receiver: {send_to_id}")

        event = _Event(id=str(uuid.uuid4()), parent_id=parent_id, from_uid=self.participant.id,
                       method_name=method_name, data=kwargs)

        return self._publish_message(event, send_to_id)

    def broadcast(self,
                  method_name: str,
                  mass_kwargs: Optional[List[Any]] = None,
                  parent_id: Optional[str] = None,
                  include_current_participant: bool = False,
                  **kwargs) -> List[ParticipantFuture]:
        self._check_if_ready()
        logger.debug("Sending event (%s) for all members" % method_name)

        if len(mass_kwargs) != len(self.members):
            raise ValueError(f"Length of arguments list ({len(mass_kwargs)}) is not equal "
                             f"to the length of members ({len(self.members)})")

        futures = []

        for args, member_id in zip(mass_kwargs, self.members):
            if member_id == self.participant.id and not include_current_participant:
                continue

            event = _Event(id=str(uuid.uuid4()), parent_id=parent_id, from_uid=self.participant.id,
                           method_name=method_name, data={'data': args, **kwargs})

            future = self._publish_message(event, member_id)
            if future:
                futures.append(future)

        return futures

    def _publish_message(self, event: _Event, receiver_id: str) -> Optional[ParticipantFuture]:
        logger.debug("Party communicator %s: sending to %s event %s" % (self.participant.id, receiver_id, event))
        # not all command requires feedback
        if event.parent_id:
            future = ParticipantFuture(participant_id=receiver_id)
            self._event_futures[event.id] = future
        else:
            future = None

        self._party_info[receiver_id].queue.put(event)

        logger.debug("Party communicator %s: sent to %s event %s" % (self.participant.id, receiver_id, event.id))

        return future

    def _check_if_ready(self):
        if not self.is_ready:
            raise RuntimeError("Cannot proceed because communicator is not ready. "
                               "Perhaps, randezvous has not been called or was unsuccessful")

    @abstractmethod
    def _run(self):
        ...


class LocalMasterPartyCommunicator(LocalPartyCommunicator):
    # todo: add docs
    def __init__(self,
                 participant: PartyMaster,
                 world_size: int,
                 shared_party_info: Optional[Dict[str, _ParticipantInfo]]):
        self.participant = participant
        self.world_size = world_size
        self._party_info: Optional[Dict[str, _ParticipantInfo]] = shared_party_info
        self._event_futures: Optional[Dict[str, Future]] = None

    def run(self):
        logger.info("Party communicator %s: running" % self.participant.id)
        self.randezvous()
        party = LocalPartyImpl(party_communicator=self)

        event_loop = Thread(name=f"event-loop-{self.participant.id}", target=self._run)
        event_loop.start()

        self.participant.run(party)

        event_loop.join()
        logger.info("Party communicator %s: finished" % self.participant.id)

    def _run(self):
        logger.info("Party communicator %s: starting event loop" % self.participant.id)

        while True:
            event = self._party_info[self.participant.id].queue.get()

            logger.debug("Party communicator %s: received event %s" % (self.participant.id, event))

            if event.method_name == _Method.service_return_answer.value:
                if event.parent_id not in self._event_futures:
                    raise ValueError(f"No awaiting future with if {event.parent_id}."
                                     f"(Event {event.id} from {event.from_uid})")

                logger.debug("Party communicator %s: marking future %s as finished by answer of event %s"
                             % (self.participant.id, event.parent_id, event.id))

                future = self._event_futures.pop(event.parent_id)
                future.set_result(event.data)
                future.done()
            elif event.method_name == _Method.service_heartbeat.value:
                logger.info("Party communicator %s: received heartbeat from %s: %s"
                            % (self.participant. id, event.id, event.data))
            elif event.method_name == _Method.finalize:
                logger.info("Party communicator %s: finalized" % self.participant.id)
                break
            else:
                raise ValueError(f"Unsupported method {event.method_name} (Event {event.id} from {event.from_uid})")

        logger.info("Party communicator %s: finished event loop" % self.participant.id)


class LocalMemberPartyCommunicator(LocalPartyCommunicator):
    # todo: add docs
    def __init__(self,
                 participant: PartyMember,
                 world_size: int,
                 shared_party_info: Optional[Dict[str, _ParticipantInfo]]):
        self.participant = participant
        self.world_size = world_size
        self._party_info: Optional[Dict[str, _ParticipantInfo]] = shared_party_info
        self._event_futures: Optional[Dict[str, Future]] = None

    def _run(self):
        logger.info("Party communicator %s: starting event loop" % self.participant.id)

        supported_methods = [
            _Method.record_uids.value,
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
                result = method(**event.data)

                self.send(send_to_id=event.from_uid, method_name=_Method.service_return_answer.value,
                          parent_id=event.id, result=result)

                if event.method_name == _Method.finalize.value:
                    break
            else:
                raise ValueError(f"Unsupported method {event.method_name} (Event {event.id} from {event.from_uid})")

        logger.info("Party communicator %s: finished event loop" % self.participant.id)


class LocalPartyImpl(Party):
    # todo: add docs
    # todo: add logging to the methods
    def __init__(self, party_communicator: PartyCommunicator, op_timeout: Optional[float] = None):
        self.party_communicator = party_communicator
        self.op_timeout = op_timeout

    def _sync_broadcast_to_members(self,
                                   method_name: _Method,
                                   mass_kwargs: Optional[List[Any]] = None, **kwargs) -> List[Any]:
        futures = self.party_communicator.broadcast(method_name=method_name.value, mass_kwargs=mass_kwargs, **kwargs)
        futures = concurrent.futures.wait(futures, timeout=self.op_timeout)
        completed_futures, uncompleted_futures \
            = cast(Set[ParticipantFuture], futures[0]), cast(Set[ParticipantFuture], futures[1])

        if len(uncompleted_futures) > 0:
            # todo: custom exception with additional info about uncompleted tasks
            raise RuntimeError(f"Not all tasks have been completed. "
                               f"Completed tasks: {len(completed_futures)}. "
                               f"Uncompleted tasks: {len(uncompleted_futures)}.")

        fresults = {future.participant_id: future.result() for future in completed_futures}
        return [fresults[member_id] for member_id in self.party_communicator.members]

    def records_uids(self) -> List[List[str]]:
        return cast(List[List[str]], self._sync_broadcast_to_members(method_name=_Method.record_uids))

    def register_records_uids(self, uids: List[str]):
        self._sync_broadcast_to_members(method_name=_Method.register_records_uids, uids=uids)

    def initialize(self):
        self._sync_broadcast_to_members(method_name=_Method.initialize)

    def finalize(self):
        self._sync_broadcast_to_members(method_name=_Method.finalize)

    def update_weights(self, upd: PartyDataTensor):
        self._sync_broadcast_to_members(method_name=_Method.update_weights, upd=upd)

    def predict(self, use_test: bool = False) -> PartyDataTensor:
        return cast(PartyDataTensor, self._sync_broadcast_to_members(method_name=_Method.predict, use_test=True))

    def update_predict(self, batch: List[str], upd: PartyDataTensor) -> PartyDataTensor:
        return cast(
            PartyDataTensor,
            self._sync_broadcast_to_members(method_name=_Method.update_predict, batch=batch, upd=PartyDataTensor)
        )
