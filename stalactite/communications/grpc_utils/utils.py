from dataclasses import dataclass, field
import enum
import logging
import pickle
from queue import Queue
from typing import Any
import concurrent

import grpc
import torch
import safetensors.torch

from stalactite.communications.grpc_utils.generated_code import communicator_pb2
from stalactite.base import ParticipantFuture
logger = logging.getLogger(__name__)


class Status(str, enum.Enum):
    not_started = 'not started'
    all_ready = 'all ready'
    waiting = 'waiting for others'
    finished = 'finished'


class ClientStatus(str, enum.Enum):
    alive = 'alive'


@dataclass
class MethodMessage:
    tensor_kwargs: dict[str, torch.Tensor] = field(default_factory=dict)
    other_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class SerializedMethodMessage:
    tensor_kwargs: dict[str, bytes] = field(default_factory=dict)
    other_kwargs: dict[str, bytes] = field(default_factory=dict)


class MessageTypes(str, enum.Enum):
    server_task = 'server'
    client_task = 'client'
    acknowledgment = 'ack'


@dataclass
class ParticipantTasks:
    context: grpc.aio.ServicerContext
    queue: Queue

@dataclass
class PreparedTask:
    task_id: str
    task_message: communicator_pb2
    task_future: ParticipantFuture


def load_data(serialized_tensor: bytes) -> torch.Tensor:
    return safetensors.torch.load(serialized_tensor)['tensor']


def save_data(tensor: torch.Tensor):
    return safetensors.torch.save(tensors={'tensor': tensor})


def prepare_kwargs(kwargs: MethodMessage | None) -> SerializedMethodMessage:
    if kwargs is None:
        return SerializedMethodMessage()
    serialized_tensors = {}
    for key, value in kwargs.tensor_kwargs.items():
        assert isinstance(value, torch.Tensor), 'MethodMessage.tensor_kwargs can contain only torch.Tensor-s as values'
        serialized_tensors[key] = save_data(value)
    other_kwargs = {}
    for key, value in kwargs.other_kwargs.items():
        if isinstance(value, torch.Tensor):
            logger.warning(
                f'Got kwarg {key} as the field in MethodMessage.other_kwargs, '
                'while it should be passed in MethodMessage.tensor_kwargs'
            )
        other_kwargs[key] = pickle.dumps(value)

    return SerializedMethodMessage(
        tensor_kwargs=serialized_tensors,
        other_kwargs=other_kwargs,
    )


def collect_kwargs(message_kwargs: SerializedMethodMessage) -> dict:
    tensor_kwargs = {}
    for key, value in message_kwargs.tensor_kwargs.items():
        tensor_kwargs[key] = load_data(value)
    other_kwargs = {}
    for key, value in message_kwargs.other_kwargs.items():
        other_kwargs[key] = pickle.loads(value)
    return {**tensor_kwargs, **other_kwargs}
