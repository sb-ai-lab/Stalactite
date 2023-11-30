from abc import ABC, abstractmethod
from typing import List, Optional

from stalactite.base import DataTensor
from stalactite.pet import PrivacyGuard

import torch


class PartyMember(ABC):
    model: torch.nn.Module = None
    privacy = None
    data: torch.Tensor

    @abstractmethod
    def records_uids(self) -> List[str]:
        ...

    @abstractmethod
    def register_records_uids(self, uids: List[str]):
        ...

    @abstractmethod
    def initialize(self, model: torch.nn.Module,
                   privacy_method: str):
        self.model = model
        if privacy_method:
            self.privacy = PrivacyGuard(privacy_method)

    @abstractmethod
    def finalize(self):
        ...

    @abstractmethod
    def update_weights(self, upd: DataTensor):

        ...

    @abstractmethod
    def predict(self, batch_ids: List[int]) -> DataTensor:
        batch = data[batch_ids]
        int_res = self.model.forward(batch)
        if self.privacy:
            int_res = self.privacy.encode_intermediate_results(int_res)
        return int_res

    @abstractmethod
    def update_predict(self, batch: List[str], upd: DataTensor) -> DataTensor:
        ...
