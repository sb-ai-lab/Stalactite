from abc import ABC, abstractmethod
from typing import List

import torch

DataTensor = torch.Tensor


class Batcher(ABC):
    uids: List[str]

    @abstractmethod
    def __iter__(self):
        ...


