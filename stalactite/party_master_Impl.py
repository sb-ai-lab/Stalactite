import random
import logging
import time
from threading import Thread
from typing import List

from party_master import PartyMaster
from stalactite.base import PartyDataTensor, DataTensor, Batcher
from stalactite.communications import Party

logger = logging.getLogger("my_logger")

class PartyMasterImpl(PartyMaster):

    def __init__(self, epochs):
        self.epochs = epochs

    def run(self, party):

        # party = self.randezvous()
        # uids = self.synchronize_uids(party)

        self.master_initialize()

        self.loop(
            batcher=self.make_batcher(uids=[0, 1, 2]),
            party=party
        )

        self.master_finalize()

    def loop(self, batcher: Batcher, party: Party):

        updates = self.make_init_updates(party.world_size)
        # начальные rhs'ы
        for epoch in range(self.epochs):
            logger.debug(f"PARTY MASTER: TRAIN LOOP - starting EPOCH {epoch}")

            for i, batch in enumerate(batcher):
                logger.debug(f"PARTY MASTER: TRAIN LOOP - starting BATCH {i}")

                party_predictions = party.update_predict(batch[0], updates)  # batch[0] is X, batch[1] in Y
                predictions = self.aggregate(party_predictions)
                updates = self.compute_updates(predictions, party_predictions, party.world_size) # todo: add Y here?
                time.sleep(10)
    def make_batcher(self, uids: List[str]) -> Batcher:
        return [(x, 1) for x in range(2)]

    def make_init_updates(self, world_size: int) -> PartyDataTensor:
        logger.debug("PARTY MASTER: making init updates")
        return [random.random() for _ in range(world_size)]

    def master_finalize(self):
        pass

    def master_initialize(self):
        pass

    def randezvous(self) -> Party:
        pass

    def report_metrics(self, y: DataTensor, predictions: DataTensor, name: str):
        pass

    def synchronize_uids(self, party: Party) -> List[str]:
        pass
    def aggregate(self, party_predictions: PartyDataTensor) -> DataTensor:
        """
        summ of all members' predictions
        :param party_predictions:
        :return:
        """
        return party_predictions

    def compute_updates(self, predictions: DataTensor, party_predictions: PartyDataTensor, world_size: int) -> List[DataTensor]:
        """
        Compute rhs's for all PartyMembers: from labels and predictions
        :param predictions:
        :param party_predictions:
        :param world_size:
        :return:
        """
        logger.debug(f"PARTY MASTER: compute_updates")

        return [random.random() for _ in range(world_size)]
