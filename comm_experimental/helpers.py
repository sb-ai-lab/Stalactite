from dataclasses import dataclass
import enum
from typing import Any, Iterator

import numpy as np
import torch


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

class ClientTask(enum.Enum):
    batched_exchange = 0
    exchange = 1



class PingResponse(str, enum.Enum):
    waiting_for_other_connections = 'waiting for other connections'
    all_ready = 'all ready'


class ClientStatus(enum.Enum):
    waiting = 0
    active = 1
    finished = 2
