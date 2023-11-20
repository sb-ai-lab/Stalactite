from abc import ABC, abstractmethod
from typing import List

import torch

DataTensor = torch.Tensor
# in reality, it will be a DataTensor but with one more dimension
PartyDataTensor = torch.Tensor


class Batcher(ABC):
    uids: List[str]

    @abstractmethod
    def __iter__(self):
        ...


