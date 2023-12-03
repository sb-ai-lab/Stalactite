import sys
import collections
import logging
from typing import List, Dict
from queue import Queue
from threading import Thread

import torch

from stalactite.base import PartyDataTensor, Party
from stalactite.communications import Event

formatter = logging.Formatter(
    fmt='(%(threadName)-9s) %(message)s',
    # datefmt='%Y-%m-%d %H:%M:%S'
)
StreamHandler = logging.StreamHandler(stream=sys.stdout)
StreamHandler.setFormatter(formatter)
logging.basicConfig(handlers=[StreamHandler], level=logging.DEBUG)
logging.getLogger("urllib3").setLevel(logging.CRITICAL)

logger = logging.getLogger("my_logger")


class PartyImpl(Party):

    def __init__(self, master, members: list):
        self.master = master
        self.members = members
        self.master_q = Queue()
        self.members_q = [Queue() for _ in members]
        self.predictions_q = Queue()
        self.party_counter = collections.Counter()

    def initialize(self):
        logger.debug("init party")
        # for _ in range(2):
        event = Event("init", 0)
        self.master_q.put(event)

    def finalize(self):
        return

    def synchronize_uids(self) -> List[str]:
        return ["a", "b", "c", "d", "e"]

    def randezvous(self) -> Party:
        pass

    def predict(self, use_test: bool = False) -> PartyDataTensor:
        """
        1. Sent prediction request to all members
        2. collect all predictions from members
        :param use_test:
        :return:
        """
        return torch.stack([torch.rand(5) for x in range(self.world_size)]) #5 is ds_size

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
        4. Collect all predictions from members and return
        """
        logger.debug("PARTY: do update predict")
        preds_dict = {}

        for i, m in enumerate(self.members):
            event = Event(type="rhs", data={"uids": batch[0], "rhs":  upd[i], "member_id": i})
            logger.debug(f"PARTY: Sending  batch & rhs to member {i+1}")
            self.members_q[i].put(event)
            self.party_counter["rhs_send"] += 1

        self.party_counter["rhs_batch_send"] += 1
        while len(preds_dict) < self.world_size:
            event = self.predictions_q.get()
            if event.type == "pred":
                member = event.data["member_id"]
                logger.debug(f"PARTY: GET prediction from member {member}")
                preds_dict[member] = event.data["prediction"]
            logger.debug(f"PARTY: predictions count: {len(preds_dict)}")
        return torch.stack([preds_dict[x] for x in range(self.world_size)])

    def member_func(self, member_q: Queue, member):
        while True:
            event = member_q.get()
            logger.debug(f"getting event from member queue with {event.type} type")

            if event.type == "123":
                self.party_counter[event.type] += 1

            elif event.type == "rhs":
                uids = event.data["uids"]
                rhs = event.data["rhs"]
                pred = member.update_predict(uids, rhs)
                member_id = event.data["member_id"]
                event = Event(type="pred", data={"prediction": pred, "member_id": member_id})
                self.predictions_q.put(event)
                logger.debug(f"Sending  preds to party")

            if self.party_counter["rhs_batch_send"] == 3:  # todo: rewrite finalise condition
                logger.debug(f"Stopping member thread...")
                break

    def run(self):
        threads = []
        for i, member in enumerate(self.members):
            th_member = Thread(target=self.member_func, args=(self.members_q[i], member))
            threads.append(th_member)
            logger.debug(f"starting member thread-{i + 2}")
            th_member.start()
            logger.debug(f"member thread-{i+2} started")

        self.master.run(self)

        for i, t in enumerate(threads):
            logger.debug(f"before joining thread {i+1}")
            t.join()
            logger.debug(f"thread {i+1} done")



