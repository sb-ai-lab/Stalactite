import collections

from abc import ABC, abstractmethod
from typing import List, Dict

from stalactite.base import DataTensor, PartyDataTensor


class Party(ABC):
    @property
    @abstractmethod
    def world_size(self) -> int:
        ...

    @abstractmethod
    def send(self, method_name: str, mass_kwargs: Dict[str, list], **kwargs):
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
    def randezvous(self):
        ...

    @abstractmethod
    def synchronize_uids(self) -> List[str]:
        uids = (uid for member_uids in self.records_uids() for uid in set(member_uids))
        shared_uids = [uid for uid, count in collections.Counter(uids).items() if count == self.world_size]

        self.register_records_uids(shared_uids)

        return shared_uids
