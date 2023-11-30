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
