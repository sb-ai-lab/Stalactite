import enum
import logging
import uuid
from concurrent.futures import Future
from dataclasses import dataclass
from queue import Queue
from threading import Thread
from typing import List, Dict, Any, Optional

from stalactite.base import PartyMaster, PartyMember, PartyCommunicator

logger = logging.getLogger(__name__)


class _Methods(enum.Enum):
    service_return_answer = 'service_return_answer'
    service_heartbeat = 'service_heartbeat'

    synchronize_uids = 'synchronize_uids'
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
class _MemberInfo:
    queue: Queue
    thread: Thread


class LocalThreadBasedPartyCommunicator(PartyCommunicator):
    def __init__(self, participant_id: str, master: PartyMaster, members: List[PartyMember]):
        if len(members) == 0:
            raise ValueError("Members list cannot be empty")

        self.participant_id = participant_id
        self.world_size = len(members)
        self.master = master
        self.members = members
        self._event_futures: Optional[Dict[str, Future]] = None
        self._participants_info: Optional[Dict[str, _MemberInfo]] = None

    @property
    def is_ready(self) -> bool:
        return self._participants_info is not None

    def randezvous(self):
        self._event_futures = dict()
        self._participants_info = {
            self.master.id: _MemberInfo(
                queue=Queue(),
                thread=Thread(name="master_event_loop", target=self._master_thread_func)
            ),
            **((m.id, _MemberInfo(
                queue=Queue(),
                thread=Thread(name="member_event_loop", target=self._member_thread_func, kwargs={"member": m}))
                ) for m in self.members)
        }

    def send(self, send_to_id: str, method_name: str, parent_id: Optional[str] = None, **kwargs) -> Future:
        self._check_if_ready()

        if send_to_id not in self._participants_info:
            raise ValueError(f"Unknown receiver: {send_to_id}")

        event = _Event(id=str(uuid.uuid4()), parent_id=parent_id, from_uid=self.participant_id,
                       method_name=method_name, data=kwargs)

        return self._publish_message(event, send_to_id)

    def broadcast_send(self, method_name: str, mass_kwargs: Dict[str, Any], **kwargs) -> List[Future]:
        self._check_if_ready()
        logger.debug("Sending event (%s) for all members" % method_name)

        futures = []

        for m in self.members:
            args = mass_kwargs.get(m.id, None)

            if not args:
                logger.warning("No data to sent for member %s. Skipping it." % m.id)
                continue

            event = _Event(id=str(uuid.uuid4()), parent_id=None, from_uid=self.participant_id,
                           method_name=method_name, data={'data': args, **kwargs})

            future = self._publish_message(event, m.id)
            futures.append(future)

        return futures

    def _publish_message(self, event: _Event, receiver_id: str):
        future = Future()
        self._event_futures[event.id] = future
        self._participants_info[receiver_id].queue.put(event)

        logger.debug(f"Sent to %s event %s" % (receiver_id, event))

        return future

    def _check_if_ready(self):
        if not self.is_ready:
            raise RuntimeError("Cannot proceed because communicator is not ready. "
                               "Perhaps, randezvous has not been called or was unsuccessful")

    def _master_thread_func(self):
        logger.info("Master thread for master %s has started" % self.master.id)

        while True:
            event = self._participants_info[self.master.id].queue.get()

            logger.debug("Received event %s" % event)

            if event.method_name == _Methods.service_return_answer.value:
                if event.parent_id not in self._event_futures:
                    raise ValueError(f"No awaiting future with if {event.parent_id}."
                                     f"(Event {event.id} from {event.from_uid})")

                logger.debug("Marking future %s as finished by answer of event %s" % (event.parent_id, event.id))
                future = self._event_futures.pop(event.parent_id)
                future.set_result(event.data)
                future.done()
            elif event.method_name == _Methods.service_heartbeat.value:
                logger.info("Received heartbeat from %s: %s" % (event.id, event.data))
            elif event.method_name == _Methods.finalize:
                logger.info("Finalized")
                break
            else:
                raise ValueError(f"Unsupported method {event.method_name} (Event {event.id} from {event.from_uid})")

        logger.info("Master thread for master %s has finished" % self.master.id)

    def _member_thread_func(self, member: PartyMember):
        logger.info("Member thread for member %s has started" % member.id)

        supported_methods = [
            _Methods.synchronize_uids.value,
            _Methods.register_records_uids.value,
            _Methods.initialize.value,
            _Methods.finalize.value,
            _Methods.update_weights.value,
            _Methods.predict.value,
            _Methods.update_predict.value
        ]

        not_found_methods = [method_name for method_name in supported_methods if hasattr(member, method_name)]
        if len(not_found_methods) > 0:
            raise RuntimeError(f"Found methods not supported by member {member.id}: {not_found_methods}")

        while True:
            event = self._participants_info[member.id].queue.get()

            logger.debug("Received event %s" % event)

            if event.method_name in supported_methods:
                method = getattr(member, event.method_name)
                result = method(**event.data)

                self.send(send_to_id=event.from_uid, method_name=_Methods.service_return_answer.value,
                          parent_id=event.id, result=result)

                if event.method_name == _Methods.finalize.value:
                    break
            else:
                raise ValueError(f"Unsupported method {event.method_name} (Event {event.id} from {event.from_uid})")

        logger.info("Member thread for member %s has finished" % member.id)
