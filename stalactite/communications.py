from abc import ABC, abstractmethod
from typing import List, Dict

from stalactite.base import DataTensor


class Party(ABC):
    @property
    @abstractmethod
    def world_size(self) -> int:
        ...

    @abstractmethod
    def send(self, method_name: str, mass_kwargs: Dict[str, list], **kwargs):
        ...

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

    # in reality, 'upd' will be a DataTensor but with one more dimension
    # now it is list for illustrative purposes
    @abstractmethod
    def update_weights(self, upd: List[DataTensor]):
        ...

    # in reality, 'upd' will be a DataTensor but with one more dimension
    # now it is list for illustrative purposes
    @abstractmethod
    def predict(self) -> List[DataTensor]:
        ...

    # in reality, 'upd' will be a DataTensor but with one more dimension
    # now it is list for illustrative purposes
    @abstractmethod
    def update_predict(self, upd: List[DataTensor]) -> List[DataTensor]:
        ...
