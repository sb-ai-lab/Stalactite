import collections
import logging
from abc import ABC, abstractmethod
from asyncio import Future
from typing import List, Dict, Any

import torch


logger = logging.getLogger(__name__)


DataTensor = torch.Tensor
# in reality, it will be a DataTensor but with one more dimension
PartyDataTensor = torch.Tensor


class Batcher(ABC):
    uids: List[str]

    @abstractmethod
    def __iter__(self):
        ...


class Party(ABC):
    world_size: int

    @abstractmethod
    def send(self, method_name: str, mass_kwargs: Dict[str, Any], **kwargs) -> List[Future]:
        ...

    @abstractmethod
    def randezvous(self):
        ...

    @abstractmethod
    def records_uids(self) -> List[str]:
        ...

    @abstractmethod
    def register_records_uids(self, uids: List[str]):
        ...

    @abstractmethod
    def initialize(self):
        ...

    @abstractmethod
    def finalize(self):
        ...

    @abstractmethod
    def update_weights(self, upd: PartyDataTensor):
        ...

    @abstractmethod
    def predict(self, use_test: bool = False) -> PartyDataTensor:
        ...

    @abstractmethod
    def update_predict(self, batch: List[str], upd: PartyDataTensor) -> PartyDataTensor:
        ...

    @abstractmethod
    def synchronize_uids(self) -> List[str]:
        uids = (uid for member_uids in self.records_uids() for uid in set(member_uids))
        shared_uids = [uid for uid, count in collections.Counter(uids).items() if count == self.world_size]

        self.register_records_uids(shared_uids)

        return shared_uids


class PartyMaster(ABC):
    id: str

    epochs: int
    report_train_metrics_iteration: int
    report_test_metrics_iteration: int
    Y: DataTensor
    epoch_counter: int
    batch_counter: int

    def run(self, party: Party):
        party.randezvous()

        uids = party.synchronize_uids()

        self.master_initialize()

        self.loop(
            batcher=self.make_batcher(uids=uids),
            party=party
        )

        self.master_finalize()

    def loop(self, batcher: Batcher, party: Party):
        updates = self.make_init_updates(party.world_size)
        for epoch in range(self.epochs):
            logger.debug(f"PARTY MASTER: TRAIN LOOP - starting EPOCH {epoch}")
            for i, batch in enumerate(batcher):
                logger.debug(f"PARTY MASTER: TRAIN LOOP - starting BATCH {i}")
                party_predictions = party.update_predict(batch, updates)
                predictions = self.aggregate(party_predictions)
                updates = self.compute_updates(predictions, party_predictions, party.world_size)

                if self.report_train_metrics_iteration > 0 and i % self.report_train_metrics_iteration == 0:
                    party_predictions = party.predict()
                    predictions = self.aggregate(party_predictions)
                    self.report_metrics(self.Y, predictions, name="Train")

                if self.report_test_metrics_iteration > 0 and i % self.report_test_metrics_iteration == 0:
                    party_predictions = party.predict(use_test=True)
                    predictions = self.aggregate(party_predictions)
                    self.report_metrics(self.Y, predictions, name="Test")
                self.batch_counter += 1
            self.epoch_counter += 1

    @abstractmethod
    def make_batcher(self, uids: List[str]) -> Batcher:
        ...

    @abstractmethod
    def master_initialize(self):
        ...

    @abstractmethod
    def master_finalize(self):
        ...

    @abstractmethod
    def make_init_updates(self, world_size: int) -> PartyDataTensor:
        ...

    @abstractmethod
    def aggregate(self, party_predictions: PartyDataTensor) -> DataTensor:
        ...

    @abstractmethod
    def compute_updates(self, predictions: DataTensor, party_predictions: PartyDataTensor, world_size: int) \
            -> List[DataTensor]:
        ...

    @abstractmethod
    def report_metrics(self, y: DataTensor, predictions: DataTensor, name: str):
        ...


class PartyMember(ABC):
    id: str

    @abstractmethod
    def records_uids(self) -> List[str]:
        ...

    @abstractmethod
    def register_records_uids(self, uids: List[str]):
        ...

    @abstractmethod
    def initialize(self):
        ...

    @abstractmethod
    def finalize(self):
        ...

    @abstractmethod
    def update_weights(self, upd: DataTensor):
        ...

    @abstractmethod
    def predict(self) -> DataTensor:
        ...

    @abstractmethod
    def update_predict(self, batch: List[str], upd: DataTensor) -> DataTensor:
        ...
