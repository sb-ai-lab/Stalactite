import enum
import logging
import uuid
from concurrent.futures import Future
from dataclasses import dataclass
from queue import Queue
from threading import Thread
from typing import List, Dict, Any, Optional

from stalactite.base import Party, PartyDataTensor, PartyMaster, PartyMember, PartyCommunicator

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
        # todo cleaning the map (in threads being created)
        self._event_futures = dict()
        # todo: correct thread initialization
        self._participants_info ={
            self.master.id: _MemberInfo(
                queue=Queue(),
                thread=Thread(name="master_event_loop", target=self._master_thread_func)
            ),
            **((m.id, _MemberInfo(
                queue=Queue(),
                thread=Thread(name="member_event_loop", target=self._member_thread_func, kwargs={"member": m}))
                ) for m in self.members)
        }

    def send(self, send_to_id: str, method_name: str, **kwargs) -> Future:
        self._check_if_ready()

        if send_to_id not in self._participants_info:
            raise ValueError(f"Unknown receiver: {send_to_id}")

        event = _Event(id=str(uuid.uuid4()), parent_id=None, from_uid=self.participant_id,
                       method_name=method_name, data=kwargs)

        return self._publish_message(event, send_to_id)

    def broadcast_send(self, method_name: str, mass_kwargs: Dict[str, Any], **kwargs) -> List[Future]:
        self._check_if_ready()
        # todo logging of mass_kwargs here

        futures = []

        for i, m in enumerate(self.members):
            args = mass_kwargs.get(m.id, None)

            if not args:
                # todo: warning
                continue

            # todo: add correct from_uid
            event = _Event(id=str(uuid.uuid4()), parent_id=None, from_uid='',
                           method_name=method_name, data={'data': args, **kwargs})

            future = self._publish_message(event, m.id)

            # todo change message
            logger.debug(f"PARTY: Sending  batch & rhs to member {i + 1}")

            futures.append(future)

        return futures

    def _publish_message(self, event: _Event, receriver_id: str):
        future = Future()
        self._event_futures[event.id] = future
        self._participants_info[receriver_id].queue.put(event)
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

            if event.method_name == _Methods.service_return_answer:
                if event.parent_id not in self._event_futures:
                    raise ValueError(f"No awaiting future with if {event.parent_id}."
                                     f"(Event {event.id} from {event.from_uid})")

                future = self._event_futures.pop(event.parent_id)
                future.set_result(event.data)
                future.done()
            elif event.method_name == _Methods.service_heartbeat:
                logger.info("Received hartbeat: %s" % event.data)
            elif event.method_name == _Methods.finalize:
                logger.info("Finalized")
                break
            else:
                raise ValueError(f"Unsupported method {event.method_name} (Event {event.id} from {event.from_uid})")

        logger.info("Master thread for master %s has finished" % self.master.id)

    def _member_thread_func(self):
        pass
