from dataclasses import dataclass
from typing import Any, Iterator, ValuesView

import numpy as np
# from prometheus_async.aio import time
import torch
import safetensors.torch

from generated_code import services_pb2
from utils.helpers import (
    Serialization,
    serialization_safetensor_time,
    serialization_proto_time,
    deserialization_safetensor_time,
    deserialization_proto_time,
)


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


def aggregate_tensors_list(values: ValuesView[torch.Tensor]) -> torch.Tensor:
    return torch.sum(torch.stack(list(values)), dim=0)


def load_data(
        data: services_pb2.TensorProto | services_pb2.SafetensorDataProto,
        serialization: Serialization
) -> torch.Tensor:
    if serialization == Serialization.safetensors:
        with serialization_safetensor_time.time():
            return safetensors.torch.load(data.data)['tensor']
    elif serialization == Serialization.protobuf:
        with serialization_proto_time.time():
            return torch.asarray(data.float_data, dtype=torch.float64).reshape(list(data.dims))

def save_data(
        tensor: torch.Tensor,
        client_id: str,
        client_iteration: int,
        serialization: Serialization,
        batch: int | None = None,
        total_batches: int | None = None,
) -> services_pb2.TensorProto | services_pb2.SafetensorDataProto:
    if serialization == Serialization.safetensors:
        with deserialization_safetensor_time.time():
            message = services_pb2.SafetensorDataProto(
                data=safetensors.torch.save(tensors={'tensor': tensor}),
                iteration=client_iteration,
                client_id=client_id,
            )
    elif serialization == Serialization.protobuf:
        with deserialization_proto_time.time():
            message = services_pb2.TensorProto(
                iteration=client_iteration,
                client_id=client_id,
                data_type=services_pb2.TensorProto.FLOAT,
            )
            message.dims.extend(tensor.shape)
            message.float_data.extend(tensor.flatten().tolist())

    if batch is not None:
        message.batch = batch
    if total_batches is not None:
        message.total_batches = total_batches
    return message
