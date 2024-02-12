import collections
import enum
import itertools
import logging
import time
from abc import ABC, abstractmethod
from concurrent.futures import Future
from dataclasses import dataclass, field
from typing import Any, Iterator, List, Optional, Union

import torch

from stalactite.communications.grpc_utils.utils import UnsupportedError

logger = logging.getLogger(__name__)

logger.setLevel(logging.DEBUG)
sh = logging.StreamHandler()
sh.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
logger.addHandler(sh)

DataTensor = torch.Tensor
# in reality, it will be a DataTensor but with one more dimension
PartyDataTensor = List[torch.Tensor]

RecordsBatch = List[str]


class Method(str, enum.Enum):  # TODO _Method the same - unify
    service_return_answer = "service_return_answer"
    service_heartbeat = "service_heartbeat"

    records_uids = "records_uids"
    register_records_uids = "register_records_uids"

    initialize = "initialize"
    finalize = "finalize"

    update_weights = "update_weights"
    predict = "predict"
    update_predict = "update_predict"


@dataclass(frozen=True)
class TrainingIteration:
    seq_num: int
    subiter_seq_num: int
    epoch: int
    batch: RecordsBatch
    previous_batch: Optional[RecordsBatch]
    participating_members: Optional[List[str]]


@dataclass
class MethodKwargs:  # TODO MethodMessage the same - unify
    """Data class holding keyword arguments for called method."""

    tensor_kwargs: dict[str, torch.Tensor] = field(default_factory=dict)
    other_kwargs: dict[str, Any] = field(default_factory=dict)


class Batcher(ABC):
    # todo: add docs
    uids: List[str]

    @abstractmethod
    def __iter__(self) -> Iterator[TrainingIteration]:
        ...


class ParticipantFuture(Future):
    def __init__(self, participant_id: str):
        super().__init__()
        self.participant_id = participant_id


@dataclass
class Task:
    method_name: str
    from_id: str
    to_id: str
    id: Optional[str] = None
    method_kwargs: Optional[MethodKwargs] = None
    result: Any = None

    @property
    def kwargs_dict(self) -> dict:
        if self.method_kwargs is not None:
            return {**self.method_kwargs.other_kwargs, **self.method_kwargs.tensor_kwargs}
        else:
            return dict()


class PartyCommunicator(ABC):
    """
    Abstract base class for communication between party members and the master.

    Used to perform communication operations (i.e. send, recv, scatter, broadcast, gather) in the VFL experiment.
    """

    world_size: int
    # todo: add docs about order guaranteeing
    members: List[str]
    arbiter: Optional[str] = None
    master: Optional[str] = None

    @abstractmethod
    def rendezvous(self) -> None:
        """ Rendezvous method to synchronize party agents. """
        ...

    @property
    @abstractmethod
    def is_ready(self) -> bool:
        """ Check if the communicator is ready for communication.

        :return: True if communicator is ready, False otherwise.
        """
        ...

    @abstractmethod
    def send(
            self,
            send_to_id: str,
            method_name: Method,
            method_kwargs: Optional[MethodKwargs] = None,
            result: Optional[Any] = None,
            **kwargs,
    ) -> Task:
        """ Send a task to a specific party agent.

        :param send_to_id: Identifier of the task receiver agent.
        :param method_name: Method name to be executed on a receiver agent.
        :param method_kwargs: Keyword arguments for the method.
        :param result: Optional result of the execution.
        :param kwargs: Optional keyword arguments.

        :return: Task to be executed by the receiving party agent.
        """
        ...

    @abstractmethod
    def recv(self, task: Task, recv_results: bool = False) -> Task:
        """Receive a task from another party agent.
        If the recv on the task is called by an agent before it sends a task, then `recv_results` must be set to False.
        If the agent sent the task and now wants to recv a response, `recv_results` must be set to True.

        :param task: Task to be received.
        :param recv_results: Flag indicating whether to receive results of previously sent task.

        :return: Received task.
        """
        ...

    @abstractmethod
    def broadcast(
            self,
            method_name: Method,
            method_kwargs: Optional[MethodKwargs] = None,
            result: Optional[Any] = None,
            participating_members: Optional[List[str]] = None,
            include_current_participant: bool = False,
            **kwargs,
    ) -> List[Task]:
        """ Broadcast a task to all participating agents.

        :param method_name: Method name to be executed on a receiver agents.
        :param method_kwargs: Keyword arguments for the method (will send the same keyword arguments to each agent).
        :param result: Optional result of the execution.
        :param participating_members: List of participating party agents identifiers.
        :param include_current_participant: Flag indicating whether to include the current participant.
        :param kwargs: Optional keyword arguments.

        :return: List of tasks to be executed by the receiving party agents.
        """
        ...

    @abstractmethod
    def scatter(
            self,
            method_name: Method,
            method_kwargs: Optional[List[MethodKwargs]] = None,
            result: Optional[Any] = None,
            participating_members: Optional[List[str]] = None,
            **kwargs,
    ) -> List[Task]:
        """ Scatter tasks to all participating agents.

        :param method_name: Method name to be executed on a receiver agents.
        :param method_kwargs: List of keyword arguments for the method w.r.t. participating members order.
        :param result: Optional result of the execution.
        :param participating_members: List of participating party agents identifiers.
        :param kwargs: Optional keyword arguments.

        :return: List of tasks to be executed by the receiving party agents.
        """
        ...

    @abstractmethod
    def gather(self, tasks: List[Task], recv_results: bool = False) -> List[Task]:
        """ Gather results from a list of tasks.

        If the gather on the tasks is called by an agent before it sends / broadcasts / scatters tasks, then
        `recv_results` must be set to False.
        If the agent sent / broadcasted / scattered the tasks and now wants to gather responses, `recv_results`
        must be set to True.

        :param tasks: List of tasks to gather results from.
        :param recv_results: Flag indicating whether to gather results of previously sent task.

        :return: List of received tasks.
        """
        ...

    @abstractmethod
    def run(self):
        """ Run the communicator.

        Perform rendezvous and wait until it is finished. Start sending, receiving and processing tasks from the
        main participant loop.
        """
        ...


class PartyAgent(ABC):
    """ Abstract base class for the party in the VFL experiment. """

    def execute_received_task(self, task: Task) -> Optional[Union[DataTensor, List[str]]]:
        """ Execute received method on the master.

        :param task: Received task to execute.
        :return: Execution result.
        """
        try:
            result = getattr(self, task.method_name)(**task.kwargs_dict)
        except AttributeError as exc:
            raise UnsupportedError(f'Method {task.method_name} is not supported on {self.id}') from exc
        return result

    def run(self, party: PartyCommunicator) -> None:
        """ Run the VFL experiment with the party agent.

        Current method should implement initialization of the party agent, launching of the main training loop,
        and finalization of the experiment.

        :param party: Communicator instance used for communication between VFL agents.
        :return: None
        """
        ...

    def loop(self, batcher: Batcher, party: PartyCommunicator) -> None:
        """ Run main training loop on the VFL agent.

        :param batcher: Batcher for creating training batches.
        :param party: Communicator instance used for communication between VFL agents.

        :return: None
        """
        ...

    @abstractmethod
    def initialize(self):
        """ Initialize the party agent. """
        ...

    @abstractmethod
    def finalize(self):
        """ Finalize the party agent. """
        ...


class PartyMaster(PartyAgent, ABC):
    """ Abstract base class for the master party in the VFL experiment. """

    id: str
    epochs: int
    report_train_metrics_iteration: int
    report_test_metrics_iteration: int
    target: DataTensor
    target_uids: List[str]
    test_target: DataTensor

    _iter_time: list[tuple[int, float]] = list()

    @property
    def train_timings(self) -> list:
        """ Return list of tuples representing iteration timings from the main loop. """
        return self._iter_time

    def synchronize_uids(self, collected_uids: list[list[str]], world_size: int) -> List[str]:
        """ Synchronize unique records identifiers across party members.

        :param collected_uids: List of lists containing unique records identifiers collected from party members.
        :param world_size: Number of party members in the experiment.

        :return: Common records identifiers among the agents used in training loop.
        """
        logger.debug("Master %s: synchronizing uids for party of size %s" % (self.id, world_size))
        uids = itertools.chain(self.target_uids, (uid for member_uids in collected_uids for uid in set(member_uids)))
        shared_uids = sorted([uid for uid, count in collections.Counter(uids).items() if count == world_size + 1])
        logger.debug("Master %s: registering shared uids f size %s" % (self.id, len(shared_uids)))
        set_shared_uids = set(shared_uids)
        uid2idx = {uid: i for i, uid in enumerate(self.target_uids) if uid in set_shared_uids}
        selected_tensor_idx = [uid2idx[uid] for uid in shared_uids]

        self.target = self.target[selected_tensor_idx]
        self.target_uids = shared_uids
        logger.debug("Master %s: record uids has been successfully synchronized")
        return shared_uids

    @abstractmethod
    def make_batcher(self, uids: List[str], party_members: List[str]) -> Batcher:
        """ Make a batcher for training.

        :param uids: List of unique identifiers of dataset records.
        :param party_members: List of party members` identifiers.

        :return: Batcher instance.
        """
        ...

    @abstractmethod
    def make_init_updates(self, world_size: int) -> PartyDataTensor:
        """ Make initial updates for party members.

        :param world_size: Number of party members.

        :return: Initial updates as a list of tensors.
        """
        ...

    @abstractmethod
    def aggregate(
            self, participating_members: List[str], party_predictions: PartyDataTensor, infer: bool = False
    ) -> DataTensor:
        """ Aggregate members` predictions.

        :param participating_members: List of participating party member identifiers.
        :param party_predictions: List of party predictions.
        :param infer: Flag indicating whether to perform inference.

        :return: Aggregated predictions.
        """
        ...

    @abstractmethod
    def compute_updates(
            self,
            participating_members: List[str],
            predictions: DataTensor,
            party_predictions: PartyDataTensor,
            world_size: int,
            subiter_seq_num: int,
    ) -> List[DataTensor]:
        """ Compute updates based on members` predictions.

        :param participating_members: List of participating party member identifiers.
        :param predictions: Model predictions.
        :param party_predictions: List of party predictions.
        :param world_size: Number of party members.
        :param subiter_seq_num: Sub-iteration sequence number.

        :return: List of updates as tensors.
        """
        ...

    @abstractmethod
    def report_metrics(self, y: DataTensor, predictions: DataTensor, name: str):
        """ Report metrics based on target values and predictions.

        :param y: Target values.
        :param predictions: Model predictions.
        :param name: Name of the dataset ("Train" or "Test").

        :return: None
        """
        ...


class PartyMember(PartyAgent, ABC):
    """ Abstract base class for the member party in the VFL experiment. """

    id: str
    master_id: str
    report_train_metrics_iteration: int
    report_test_metrics_iteration: int
    _iter_time: list[tuple[int, float]] = list()

    @property
    @abstractmethod
    def batcher(self) -> Batcher:
        """ Get the batcher for training.

        :return: Batcher instance.
        """
        ...

    @abstractmethod
    def records_uids(self) -> List[str]:
        """ Get the list of existing dataset unique identifiers.

        :return: List of unique identifiers.
        """
        ...

    @abstractmethod
    def register_records_uids(self, uids: List[str]):
        """ Register unique identifiers to be used.

        :param uids: List of unique identifiers.
        :return: None
        """
        ...

    @abstractmethod
    def update_weights(self, uids: RecordsBatch, upd: DataTensor):
        """ Update model weights based on input features and target values.

        :param uids: Batch of record unique identifiers.
        :param upd: Updated model weights.
        """
        ...

    @abstractmethod
    def predict(self, uids: RecordsBatch, use_test: bool = False) -> DataTensor:
        """ Make predictions using the initialized model.

        :param uids: Batch of record unique identifiers.
        :param use_test: Flag indicating whether to use the test data.

        :return: Model predictions.
        """
        ...

    @abstractmethod
    def update_predict(self, upd: DataTensor, previous_batch: RecordsBatch, batch: RecordsBatch) -> DataTensor:
        """ Update model weights and make predictions.

        :param upd: Updated model weights.
        :param previous_batch: Previous batch of record unique identifiers.
        :param batch: Current batch of record unique identifiers.

        :return: Model predictions.
        """
        ...
