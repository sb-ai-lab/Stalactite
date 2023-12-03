import logging
import uuid
from concurrent.futures import Future
from dataclasses import dataclass
from queue import Queue
from threading import Thread
from typing import List, Dict, Any, Optional

from stalactite.base import Party, PartyDataTensor, PartyMaster, PartyMember


logger = logging.getLogger(__name__)


@dataclass
class Event:
    id: str
    parent_id: Optional[str]
    from_uid: str
    method_name: str
    data: Any


class LocalThreadBasedParty(Party):
    def __init__(self, master: PartyMaster, members: List[PartyMember]):
        if len(members) == 0:
            raise ValueError("Members list cannot be empty")

        self.world_size = len(members)
        self.master = master
        self.members = members
        self._event_futures: Optional[Dict[str, Future]] = None
        self._master_queue: Optional[Queue] = None
        self._member_queues: Optional[Dict[str, Queue]] = None
        self._member_threads: Optional[Dict[str, Thread]] = None

    def randezvous(self):
        self._event_futures = dict()
        self._master_queue = Queue()
        self._member_queues = {m.id: Queue() for m in self.members}
        # todo: correct thread initialization
        self._member_threads = {m.id: Thread() for m in self.members}

    def send(self, method_name: str, mass_kwargs: Dict[str, Any], **kwargs) -> List[Future]:
        # todo: ensure randezvous has happened
        # todo logging of mass_kwargs here

        futures = []

        for i, m in enumerate(self.members):
            args = mass_kwargs.get(m.id, None)

            if not args:
                # todo: warning
                continue

            event = Event(id=str(uuid.uuid4()), parent_id=None, from_uid='', method_name=method_name, data=args)

            future = Future()
            self._event_futures[event.id] = future

            self._member_queues[m.id].put(event)

            # todo change message
            logger.debug(f"PARTY: Sending  batch & rhs to member {i + 1}")

            futures.append(future)

        return futures

    def records_uids(self) -> List[str]:
        pass

    def register_records_uids(self, uids: List[str]):
        pass

    def initialize(self):
        pass

    def finalize(self):
        pass

    def update_weights(self, upd: PartyDataTensor):
        pass

    def predict(self, use_test: bool = False) -> PartyDataTensor:
        pass

    def update_predict(self, batch: List[str], upd: PartyDataTensor) -> PartyDataTensor:
        pass



    def synchronize_uids(self) -> List[str]:
        pass
