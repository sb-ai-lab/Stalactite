from threading import Thread
from typing import List

from party_master import PartyMaster
from stalactite.base import PartyDataTensor, DataTensor


class PartyMasterImpl(PartyMaster):
    def __init__(self, queue):
        super().__init__()
        self.members_predictions = {}
        self.members_rhs = {}
        self.queue = queue

    def run(self):
        pass

    def aggregate(self, party_predictions: PartyDataTensor) -> DataTensor:
        """
        summ of all members' predictions
        :param party_predictions:
        :return:
        """

    def compute_updates(self, predictions: DataTensor, party_predictions: PartyDataTensor, world_size: int) -> List[DataTensor]:
        """
        Compute rhs's for all PartyMembers: from labels and predictions
        :param predictions:
        :param party_predictions:
        :param world_size:
        :return:
        """

