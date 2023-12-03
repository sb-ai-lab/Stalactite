import random
import logging
from typing import List

import torch

from stalactite.base import PartyDataTensor, DataTensor, Batcher, PartyMaster

logger = logging.getLogger("my_logger")


class PartyMasterImpl(PartyMaster):
    def __init__(self, epochs: int, report_train_metrics_iteration: int, report_test_metrics_iteration: int, Y: DataTensor):
        self.epochs = epochs
        self.report_train_metrics_iteration = report_train_metrics_iteration
        self.report_test_metrics_iteration = report_test_metrics_iteration
        self.Y = Y
        self.epoch_counter = 0
        self.batch_counter = 0

    def make_batcher(self, uids: List[str]) -> Batcher:
        """
        ['a', 'b', 'c', 'd', 'e'] - uids
        [0, 0, 1, 1, 0] - targets
        (['a', 'b'], [0, 0]) - batch1
        (['c', 'd'], [1, 1]) batch2
        (['e'], [0]) - batch3

        :param uids:
        :return:
        """
        batch_size = 2  # todo: this must be in input
        y_list = [random.randint(0, 1) for u in uids]  # todo: this must be in input
        return [(uids[pos:pos + batch_size], y_list[pos:pos + batch_size]) for pos in range(0, len(uids), batch_size)] #todo: do this in Batcherimpl

    def make_init_updates(self, world_size: int) -> PartyDataTensor:
        logger.debug("PARTY MASTER: making init updates")
        return torch.rand(world_size)

    def master_finalize(self):
        pass

    def master_initialize(self):
        pass

    def report_metrics(self, y: DataTensor, predictions: DataTensor, name: str):
        pass

    def aggregate(self, party_predictions: PartyDataTensor) -> DataTensor:
        """
        summ of all members' predictions
        :param party_predictions:
        :return:
        """
        return torch.sum(party_predictions, dim=1)

    def compute_updates(self, predictions: DataTensor, party_predictions: PartyDataTensor, world_size: int) -> List[DataTensor]:
        """
        Compute rhs's for all PartyMembers: from labels and predictions
        :param predictions:
        :param party_predictions:
        :param world_size:
        :return:
        """
        logger.debug(f"PARTY MASTER: compute_updates")

        return [torch.rand(1) for _ in range(predictions.size(dim=0))]
