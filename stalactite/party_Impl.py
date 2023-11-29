import sys
import collections
import logging
import random
import time
from typing import List, Any, Dict
from dataclasses import dataclass
from queue import Queue
from threading import Thread

from communications import Party
from stalactite.base import PartyDataTensor

# logging.basicConfig(level=logging.DEBUG,
#                     format='(%(threadName)-9s) %(message)s',)


formatter = logging.Formatter(
    fmt='(%(threadName)-9s) %(message)s',
    # datefmt='%Y-%m-%d %H:%M:%S'
)
StreamHandler = logging.StreamHandler(stream=sys.stdout)
StreamHandler.setFormatter(formatter)
logging.basicConfig(handlers=[StreamHandler], level=logging.DEBUG)
logging.getLogger("urllib3").setLevel(logging.CRITICAL)

logger = logging.getLogger("my_logger")

@dataclass
class Event:
    type: str
    data: Any


class PartyImpl(Party):

    def __init__(self, master, members: list):
        self.master = master
        self.members = members
        self.master_q = Queue()
        self.members_q = [Queue() for _ in members]
        self.party_counter = collections.Counter()
        self.preds = [] #move it to master

    def initialize(self):
        logger.debug("init party")
        # for _ in range(2):
        event = Event("init", 0)
        self.master_q.put(event)

    def finalize(self):
        return

    def predict(self, use_test: bool = False) -> PartyDataTensor:
        pass

    def records_uids(self) -> List[str]:
        pass

    def register_records_uids(self, uids: List[str]):
        pass

    def update_weights(self, upd: PartyDataTensor):
        pass

    def send(self, method_name: str, mass_kwargs: Dict[str, list], **kwargs):
        pass

    @property
    def world_size(self) -> int:
        return len(self.members)

    def update_predict(self, batch: List[str], upd: PartyDataTensor) -> PartyDataTensor:
        """
        1. Sent rhs's from master to all members
        2. Update weights in members side
        3. Make batch predictions on members
        4. Sent predictions from all members to master
        """

        logger.debug("PARTY: do update predict")
        for i, m in enumerate(self.members):
            event = Event(type="rhs", data={"X": batch, "rhs":  upd[i]})
            logger.debug(f"PARTY: Sending  batch & rhs to member {i+1}")
            self.members_q[i].put(event)

        self.party_counter["rhs_send"] += 1

    def Master_func(self, master_q: Queue):
        while True:
            event = master_q.get(timeout=5)
            logger.debug(f"getting event from master queue with {event.type} type")
            # if event.type == "initialised":
            #     self.party_counter[event.type] += 1
            # if event.type == "pred":
            #     self.preds.append(event.data)  #
            # # self.update_predict(batch=[], upd=0)
            #
            logger.debug(f"party_counter: rhs send {self.party_counter['rhs_send']}")

            if self.party_counter["rhs_send"] == 2: #2 is batch size
                logger.debug(f"Stopping master thread...")
                break
            time.sleep(5)

    def member_func(self, member_q: Queue, member):
        while True:
            event = member_q.get()
            logger.debug(f"getting event from member queue with {event.type} type")

            if event.type == "123":
                self.party_counter[event.type] += 1

            elif event.type == "rhs":
                batch = event.data["X"]
                rhs = event.data["rhs"]
                pred = member.update_predict(batch, rhs)
                event = Event(type="pred", data=pred)
                self.master_q.put(event)
                logger.debug(f"Sending  preds to master")

            if self.party_counter["rhs_send"] == 2:
                logger.debug(f"Stopping member thread...")
                break
            time.sleep(5)

    def run(self):
        threads = []
        th_master = Thread(target=self.Master_func, args=(self.master_q,))
        threads.append(th_master)
        logger.debug("starting master-1 thread")
        th_master.start()
        logger.debug("master thread-1 started")
        for i, member in enumerate(self.members):
            th_member = Thread(target=self.member_func, args=(self.members_q[i], member))
            threads.append(th_member)
            logger.debug(f"starting member thread-{i + 2}")
            th_member.start()
            logger.debug(f"member thread-{i+2} started")

        self.master.run(self) #???

        for i, t in enumerate(threads):
            logger.debug(f"before joining thread {i+1}")
            t.join()
            logger.debug(f"thread {i+1} done")



