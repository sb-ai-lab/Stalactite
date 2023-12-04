import logging
import random
from typing import List, Optional

import torch

from stalactite.base import PartyMaster, DataTensor, Batcher, PartyDataTensor, PartyMember


logger = logging.getLogger(__name__)


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


class MockPartyMemberImpl(PartyMember):
    def __init__(self):
        self._uids = [str(i) for i in range(100)]
        self._uids_to_use: Optional[List[str]] = None
        self._is_initialized = False
        self._is_finalized = False
        self._weights: Optional[DataTensor] = None

    def records_uids(self) -> List[str]:
        return self._uids

    def register_records_uids(self, uids: List[str]):
        self._uids_to_use = uids

    def initialize(self):
        self._weights = torch.rand()
        self._is_initialized = True

    def finalize(self):
        self._check_if_ready()
        self._is_finalized = True

    def update_weights(self, upd: DataTensor):
        logger.debug(f"PARTY MEMBER: updating weights...")
        # todo: add batch here (uid part)? Q Nik

    def predict(self) -> DataTensor:
        logger.debug(f"PARTY MEMBER: making predict...")
        return torch.rand(len(batch))

    def update_predict(self, batch: List[str], upd: DataTensor) -> DataTensor:
        self.update_weights(upd)
        return self.predict(batch)

    def _check_if_ready(self):
        if not self._is_initialized:
            raise RuntimeError("The member has not been initialized")