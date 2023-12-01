import collections
from abc import ABC, abstractmethod
from typing import List

from stalactite.base import Batcher, DataTensor, PartyDataTensor
from stalactite.communications import Party


class PartyMaster(ABC):
    epochs: int
    report_train_metrics_iteration: int
    report_test_metrics_iteration: int
    Y: DataTensor
    lr: float

    def run(self):
        party = self.randezvous()
        uids = self.synchronize_uids(party)

        self.master_initialize()

        self.loop(
            batcher=self.make_batcher(uids),
            party=party
        )

        self.master_finalize()

    @abstractmethod
    def randezvous(self) -> Party:
        ...

    @abstractmethod
    def synchronize_uids(self, party: Party) -> List[str]:
        uids = (uid for member_uids in party.records_uids() for uid in set(member_uids))
        shared_uids = [uid for uid, count in collections.Counter(uids).items() if count == party.world_size]

        party.register_records_uids(shared_uids)

        return shared_uids

    def loop(self, batcher: Batcher, party: Party):
        updates = self.make_init_updates(party.world_size)
        for epoch in range(self.epochs):
            for i, batch in enumerate(batcher):
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

        ...

    @abstractmethod
    def make_batcher(self, uids: List[str]) -> Batcher:
        ...

    @abstractmethod
    def master_initialize(self, privacy: str):
        self.privacy = privacy
        ...

    @abstractmethod
    def master_finalize(self):
        ...

    @abstractmethod
    def make_init_updates(self, world_size: int) -> PartyDataTensor:
        ...

    @abstractmethod
    def aggregate(self, party_predictions: PartyDataTensor) -> DataTensor:
        for ix,  result in enumerate(party_predictions):
            pass
        ...

    @abstractmethod
    def compute_updates(self, predictions: DataTensor, party_predictions: PartyDataTensor, world_size: int) \
            -> List[DataTensor]:

        ...

    @abstractmethod
    def report_metrics(self, y: DataTensor, predictions: DataTensor, name: str):
        ...
