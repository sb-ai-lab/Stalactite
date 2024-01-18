import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
import enum
import logging
import pickle
from queue import Queue
from typing import Any, Optional

import grpc
import torch
import safetensors.torch
from prometheus_client import Gauge, Summary

from stalactite.communications.grpc_utils.generated_code import communicator_pb2
from stalactite.base import ParticipantFuture

logger = logging.getLogger(__name__)


class MultiElementQueue:
    _lock = threading.Lock()
    elements = list()

    def put(self, value):
        with self._lock:
            self.elements.append(value)

    def get(self):
        with self._lock:
            self.elements.pop()

    def get_all(self):
        with self._lock:
            values = self.elements
            self.elements = list()
        return values


class PrometheusMetric(enum.Enum):
    number_of_connected_agents = Gauge('number_of_connected_agents', 'Active clients number', ['experiment_label'])
    execution_time = Summary(
        'member_task_execution_time',
        'Execution time in sec of the tasks on members',
        ['experiment_label', 'client_id', 'task_type']
    )
    message_size = Summary(
        'member_task_message_size',
        'Size of the returned by member message containing results',
        ['experiment_label', 'client_id', 'task_type']
    )
    send_client_time = Summary(
        'member_send_time',
        'Time of the unary send operation to master until an acknowledgment is received',
        ['experiment_label', 'client_id', 'task_type']
    )
    iteration_times = Summary(
        'master_iteration_time',
        'Time of iterations in the training loop',
        ['experiment_label', 'iteration']
    )



class UnsupportedError(Exception):
    def __init__(self, message: str = "Unsupported method for class."):
        super().__init__(message)


class ArbiterServerError(Exception):
    def __init__(self, message: str = "Arbiter server could not process the request."):
        super().__init__(message)


@contextmanager
def start_thread(*args, thread_timeout: float = 100., **kwargs):
    thread = threading.Thread(*args, **kwargs)
    try:
        thread.start()
        yield thread
    finally:
        thread.join(timeout=thread_timeout)


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
    """
    Serialize torch.Tensor data to use it in protobuf message.

    :param serialized_tensor: Tensor to serialize
    """
    return safetensors.torch.load(serialized_tensor)['tensor']


def save_data(tensor: torch.Tensor):
    """
    Deserialize torch.Tensor from bytes.

    :param tensor: Tensor serialized with load_data function
    """
    return safetensors.torch.save(tensors={'tensor': tensor})


def prepare_kwargs(kwargs: Optional[MethodMessage]) -> SerializedMethodMessage:
    """
    Serialize data fields for protobuf message.

    :param kwargs: MethodMessage containing Task data to be serialized
    """
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
    """
    Collect and deserialize protobuf message data fields.

    :param message_kwargs: SerializedMethodMessage containing Task data after serialization
    """
    tensor_kwargs = {}
    for key, value in message_kwargs.tensor_kwargs.items():
        tensor_kwargs[key] = load_data(value)
    other_kwargs = {}
    for key, value in message_kwargs.other_kwargs.items():
        other_kwargs[key] = pickle.loads(value)
    return {**tensor_kwargs, **other_kwargs}
