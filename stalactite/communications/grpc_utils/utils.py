import enum
import logging
import pickle
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Optional, Tuple

import numpy as np
import safetensors.torch
import torch
from prometheus_client import Gauge, Histogram

from stalactite.base import MethodKwargs, ParticipantFuture
from stalactite.communications.grpc_utils.generated_code import communicator_pb2

logger = logging.getLogger(__name__)


class PrometheusMetric(enum.Enum):
    """Class holding Prometheus metrics."""

    number_of_connected_agents = Gauge(
        "number_of_connected_agents",
        "Active clients number",
        ["experiment_label", "run_id"]
    )
    execution_time = Histogram(
        "agent_task_execution_time",
        "Execution time in sec of the tasks on agents",
        ["experiment_label", "client_id", "task_type", "run_id"],
        buckets=np.concatenate([np.arange(0, 2., 0.2), np.arange(5., 120., 25.)])
    )
    recv_message_size = Histogram(
        "task_message_size",
        "Size of the received by master messages (bytes)",
        ["experiment_label", "client_id", "task_type", "run_id"],
        buckets=np.concatenate([np.arange(0, 10000, 2500), np.array([10 ** pow for pow in range(4, 9)])])
    )
    send_client_time = Histogram(
        "member_send_time",
        "Time of the unary send operation to master until an acknowledgment is received",
        ["experiment_label", "client_id", "task_type", "run_id"],
        buckets=np.concatenate([np.arange(0, 5., 0.5), np.arange(5., 35., 10.)])
    )
    iteration_time_hist = Histogram(
        "iteration_time_hist",
        "Time of the iteration in the master training loop",
        ["experiment_label", "run_id"],
        buckets=np.concatenate([np.arange(0, 3., 0.5), np.arange(5., 300., 20.)])
    )
    iteration_time_gauge = Gauge(
        "iteration_time_gauge",
        "Time of the iteration in the master training loop",
        ["experiment_label", "run_id"],
    )


class ArbiterServerError(Exception):
    """Custom exception class for errors related to the Arbiter server."""

    def __init__(self, message: str = "Arbiter server could not process the request."):
        super().__init__(message)


@contextmanager
def start_thread(*args, thread_timeout: float = 100.0, **kwargs):
    """Helper context manager to manage threads."""
    thread = threading.Thread(*args, **kwargs)
    try:
        thread.start()
        yield thread
    finally:
        thread.join(timeout=thread_timeout)


class Status(str, enum.Enum):
    """Enum representing different communicator world statuses."""

    not_started = "not started"
    all_ready = "all ready"
    waiting = "waiting for others"
    finished = "finished"


class ClientStatus(str, enum.Enum):
    """Enum representing different client statuses."""

    alive = "alive"


@dataclass
class SerializedMethodMessage:
    """Data class holding serialization keyword arguments for deserialization."""

    tensor_kwargs: dict[str, bytes] = field(default_factory=dict)
    other_kwargs: dict[str, bytes] = field(default_factory=dict)
    prometheus_kwargs: dict[str, bytes] = field(default_factory=dict)


class MessageTypes(str, enum.Enum):
    """Enum representing different message types."""

    server_task = "server"
    client_task = "client"
    acknowledgment = "ack"


@dataclass
class PreparedTask:
    """Helper data class holding Task info and future."""

    task_id: str
    task_message: communicator_pb2.MainMessage
    task_future: Optional[ParticipantFuture] = field(default=None)


def load_data(serialized_tensor: bytes) -> torch.Tensor:
    """
    Serialize torch.Tensor data to use it in protobuf message.

    :param serialized_tensor: Tensor to serialize
    """
    data = safetensors.torch.load(serialized_tensor)
    tensor = data["tensor"]
    tensor.requires_grad = bool(data['requires_grad'][0])
    return tensor


def save_data(tensor: torch.Tensor):
    """
    Deserialize torch.Tensor from bytes.

    :param tensor: Tensor serialized with load_data function
    """

    return safetensors.torch.save(
        tensors={
            "tensor": tensor,
            'requires_grad': torch.tensor([int(tensor.requires_grad)])
        }
    )


def prepare_kwargs(
        kwargs: Optional[MethodKwargs], prometheus_metrics: Optional[dict] = None
) -> SerializedMethodMessage:
    """
    Serialize data fields for protobuf message.

    :param kwargs: MethodMessage containing Task data to be serialized
    """
    serialized_tensors = {}
    other_kwargs = {}
    prometheus_kwargs = {}

    if kwargs is not None:
        for key, value in kwargs.tensor_kwargs.items():
            assert isinstance(
                value, torch.Tensor
            ), "MethodMessage.tensor_kwargs can contain only torch.Tensor-s as values"
            serialized_tensors[key] = save_data(value)

        for key, value in kwargs.other_kwargs.items():
            if isinstance(value, torch.Tensor):
                logger.warning(
                    f"Got kwarg: `{key}` as the field in MethodMessage.other_kwargs, "
                    "while it should be passed in MethodMessage.tensor_kwargs. "
                    "It might slow down the communication between agents."
                )
            other_kwargs[key] = pickle.dumps(value)

    if prometheus_metrics is not None:
        for key, value in prometheus_metrics.items():
            prometheus_kwargs[key] = pickle.dumps(value)
    return SerializedMethodMessage(
        tensor_kwargs=serialized_tensors,
        other_kwargs=other_kwargs,
        prometheus_kwargs=prometheus_kwargs,
    )


def collect_kwargs(
        message_kwargs: SerializedMethodMessage, prometheus_metrics: Optional[dict[str, bytes]] = None
) -> Tuple[MethodKwargs, dict, Any]:
    """
    Collect and deserialize protobuf message data fields.

    :param message_kwargs: SerializedMethodMessage containing Task data after serialization
    """
    result = None

    tensor_kwargs = {}
    for key, value in message_kwargs.tensor_kwargs.items():
        tensor_kwargs[key] = load_data(value)
    other_kwargs = {}
    for key, value in message_kwargs.other_kwargs.items():
        other_kwargs[key] = pickle.loads(value)
    prometheus_kwargs = {}
    if prometheus_metrics is not None:
        for key, value in prometheus_metrics.items():
            prometheus_kwargs[key] = pickle.loads(value)

    if "result" in tensor_kwargs:
        result = tensor_kwargs.pop("result")
    elif "result" in other_kwargs:
        result = other_kwargs.pop("result")

    return MethodKwargs(tensor_kwargs=tensor_kwargs, other_kwargs=other_kwargs), prometheus_kwargs, result
