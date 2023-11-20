from abc import ABC, abstractmethod
from typing import List

from stalactite.base import DataTensor


class PartyMember(ABC):
    @abstractmethod
    def records_uuids(self) -> List[str]:
        ...

    @abstractmethod
    def register_records_uuids(self, uuids: List[str]):
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
    def update_predict(self, upd: DataTensor) -> DataTensor:
        ...
