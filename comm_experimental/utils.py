from dataclasses import dataclass
from typing import Any, Iterator

import numpy as np
import torch
import safetensors.torch



@dataclass
class BatchedData:
    data: torch.Tensor | list[Any]
    batch: int
    total_batches: int


def batch_generator(data: torch.Tensor | list[Any], batch_size: int) -> Iterator[BatchedData]:
    total_batches = int(np.ceil(len(data) / batch_size))
    for batch in range(total_batches):
        yield BatchedData(
            data=data[batch * batch_size: (batch + 1) * batch_size],
            batch=batch,
            total_batches=total_batches
        )

def load_data(data: bytes) -> torch.Tensor:
    return safetensors.torch.load(data)['tensor']


def save_data(data: torch.Tensor) -> bytes:
    return safetensors.torch.save(tensors={'tensor': data})



