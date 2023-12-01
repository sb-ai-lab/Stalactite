from abc import ABC, abstractmethod
from typing import List, Optional

from stalactite.base import DataTensor
from stalactite.pet import PrivacyGuard

import torch


class PartyMember(ABC):
    model: torch.nn.Module = None
    privacy = None
    lr: float
    public_key: DataTensor

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
        self.model.initialize(data_shape)

        if privacy_method:
            self.privacy = PrivacyGuard(privacy_method)
            self.privacy.set_public_key(self.public_key)

    def create_batch(self, batch_ids: List[str]) -> DataTensor:
        ...

    @abstractmethod
    def finalize(self):
        ...

    @abstractmethod
    def update_weights(self, upd: DataTensor):
        back_grad = self.model.backward(upd)
        self.model.step()

    @abstself.ractmethod
    def predict(self, batch_ids: List[str]) -> DataTensor:
        batch = self.create_batch(batch_ids)
        int_res = self.model.forward(batch)
        if self.privacy:
            int_res = self.privacy.secure_result(int_res)
        return int_res

    @abstractmethod
    def update_predict(self, batch_ids: List[str], upd: DataTensor) -> DataTensor:
        self.update_weights(upd)
        result = self.predict(batch_ids)
        return result
