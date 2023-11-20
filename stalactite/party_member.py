from abc import ABC, abstractmethod
from typing import List

from stalactite.base import DataTensor


class PartyMember(ABC):
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
