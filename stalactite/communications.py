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
from typing import List, Dict, Any, Optional, Union, cast

from stalactite.base import PartyMaster, PartyMember, PartyCommunicator, Party, PartyDataTensor

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
    thread: Thread


class LocalThreadBasedPartyCommunicator(PartyCommunicator):
    # todo: add docs
    # todo: introduce the single interface for that
    participant: Union[PartyMaster, PartyMember]
    _party_info: Optional[Dict[str, _ParticipantInfo]]
    _event_futures: Optional[Dict[str, Future]]

    @property
    def is_ready(self) -> bool:
        return self._party_info is not None

    def run(self):
        # todo: incorrect implementation
        self.randezvous()

    def send(self, send_to_id: str, method_name: str, parent_id: Optional[str] = None, **kwargs) -> Future:
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
                  **kwargs) -> List[Future]:
        self._check_if_ready()
        logger.debug("Sending event (%s) for all members" % method_name)

        futures = []

        for member_id in self._party_info.keys():
            if member_id == self.participant.id and not include_current_participant:
                continue

            args = mass_kwargs.get(member_id, dict()) if mass_kwargs else dict()

            event = _Event(id=str(uuid.uuid4()), parent_id=parent_id, from_uid=self.participant.id,
                           method_name=method_name, data={'data': args, **kwargs})

            future = self._publish_message(event, member_id)
            if future:
                futures.append(future)

        return futures

    def _publish_message(self, event: _Event, receiver_id: str) -> Optional[Future]:
        # not all command requires feedback
        if event.parent_id:
            future = Future()
            self._event_futures[event.id] = future
        else:
            future = None

        self._party_info[receiver_id].queue.put(event)

        logger.debug(f"Sent to %s event %s" % (receiver_id, event))

        return future

    def _check_if_ready(self):
        if not self.is_ready:
            raise RuntimeError("Cannot proceed because communicator is not ready. "
                               "Perhaps, randezvous has not been called or was unsuccessful")

    @abstractmethod
    def _thread_func(self):
        ...


class MasterLocalThreadBasedPartyCommunicator(LocalThreadBasedPartyCommunicator):
    # todo: add docs
    def __init__(self,
                 participant: PartyMaster,
                 world_size: int,
                 shared_party_info: Optional[Dict[str, _ParticipantInfo]]):
        self.participant = participant
        self.world_size = world_size
        self._party_info: Optional[Dict[str, _ParticipantInfo]] = shared_party_info
        self._event_futures: Optional[Dict[str, Future]] = None

    def randezvous(self):
        self._party_info[self.participant.id] = _ParticipantInfo(
            queue=Queue(),
            thread=Thread(name="master_event_loop", target=self._thread_func)
        )

        # todo: allow to work with timeout for randezvous operation
        while len(self._party_info) < self.world_size:
            time.sleep(0.1)

    def _thread_func(self):
        logger.info("Master thread for master %s has started" % self.participant.id)

        while True:
            event = self._party_info[self.participant.id].queue.get()

            logger.debug("Received event %s" % event)

            if event.method_name == _Method.service_return_answer.value:
                if event.parent_id not in self._event_futures:
                    raise ValueError(f"No awaiting future with if {event.parent_id}."
                                     f"(Event {event.id} from {event.from_uid})")

                logger.debug("Marking future %s as finished by answer of event %s" % (event.parent_id, event.id))
                future = self._event_futures.pop(event.parent_id)
                future.set_result(event.data)
                future.done()
            elif event.method_name == _Method.service_heartbeat.value:
                logger.info("Received heartbeat from %s: %s" % (event.id, event.data))
            elif event.method_name == _Method.finalize:
                logger.info("Finalized")
                break
            else:
                raise ValueError(f"Unsupported method {event.method_name} (Event {event.id} from {event.from_uid})")

        logger.info("Master thread for master %s has finished" % self.participant.id)


class MemberLocalThreadBasedPartyCommunicator(LocalThreadBasedPartyCommunicator):
    # todo: add docs
    def __init__(self,
                 participant: PartyMember,
                 world_size: int,
                 shared_party_info: Optional[Dict[str, _ParticipantInfo]]):
        self.participant = participant
        self.world_size = world_size
        self._party_info: Optional[Dict[str, _ParticipantInfo]] = shared_party_info
        self._event_futures: Optional[Dict[str, Future]] = None

    def randezvous(self):
        self._event_futures = dict()
        self._party_info[self.participant.id] = _ParticipantInfo(
            queue=Queue(),
            thread=Thread(name=f"member_event_loop_{self.participant.id}", target=self._thread_func)
        )

    def _thread_func(self):
        logger.info("Member thread for member %s has started" % self.participant.id)

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

            logger.debug("Received event %s" % event)

            if event.method_name in supported_methods:
                method = getattr(self.participant, event.method_name)
                result = method(**event.data)

                self.send(send_to_id=event.from_uid, method_name=_Method.service_return_answer.value,
                          parent_id=event.id, result=result)

                if event.method_name == _Method.finalize.value:
                    break
            else:
                raise ValueError(f"Unsupported method {event.method_name} (Event {event.id} from {event.from_uid})")

        logger.info("Member thread for member %s has finished" % self.participant.id)


class PartyImpl(Party):
    party_communicator: PartyCommunicator
    op_timeout: Optional[float]

    def _sync_broadcast_to_members(self,
                                   method_name: _Method,
                                   mass_kwargs: Optional[List[Any]] = None, **kwargs) -> List[Any]:
        futures = self.party_communicator.broadcast(method_name=method_name.value, mass_kwargs=mass_kwargs, **kwargs)
        completed_futures, uncompleted_futures = concurrent.futures.wait(futures, timeout=self.op_timeout)

        if len(uncompleted_futures) > 0:
            # todo: custom exception with additional info about uncompleted tasks
            raise RuntimeError(f"Not all tasks have been completed. "
                               f"Completed tasks: {len(completed_futures)}. "
                               f"Uncompleted tasks: {len(uncompleted_futures)}.")

        fresults = {future.member_id: future.result() for future in completed_futures}
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
