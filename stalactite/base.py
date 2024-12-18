import collections
import itertools
import json
import logging
import os
import time
from abc import ABC, abstractmethod
from concurrent.futures import Future
from dataclasses import dataclass
from typing import Any, Iterator, List, Optional, Union, Tuple, Dict

import torch

from stalactite.communications.helpers import MethodKwargs, Method

logger = logging.getLogger(__name__)

DataTensor = torch.Tensor
# in reality, it will be a DataTensor but with one more dimension
PartyDataTensor = List[torch.Tensor]

RecordsBatch = List[str]


@dataclass
class TaskExecutionTime:
    client_id: str
    task_name: Union[str, Method]
    execution_time: float


@dataclass
class IterationTime:
    client_id: str
    iteration: int
    iteration_time: float


@dataclass(frozen=True)
class TrainingIteration:
    seq_num: int
    subiter_seq_num: int
    epoch: int
    batch: RecordsBatch
    previous_batch: Optional[RecordsBatch]
    participating_members: Optional[List[str]]
    last_batch: bool


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
            result: Optional[Union[Any, List[Any]]] = None,
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
    is_initialized: bool = False
    is_finalized: bool = False
    id: str
    do_train: bool
    do_predict: bool
    do_save_model: bool
    _model: Union[torch.nn.Module, List[torch.nn.Module]]
    model_path: str
    uid2tensor_idx: Dict[Any, int]
    uid2tensor_idx_test: Dict[Any, int]

    task_execution_times: List[TaskExecutionTime] = list()

    @abstractmethod
    def make_batcher(
            self,
            uids: Optional[List[str]] = None,
            party_members: Optional[List[str]] = None,
            is_infer: bool = False
    ) -> Batcher:
        """ Make a make_batcher for training or inference.

        :param uids: List of unique identifiers of dataset records.
        :param party_members: List of party members` identifiers.
        :param is_infer: Flag indicating whether to use inference mode.
         Inference mode means that the batcher will have only one epoch and use eval_batch_size.

        :return: Batcher instance.
        """
        ...

    def check_if_ready(self):
        """ Raise a RuntimeError if the agent has not been initialized or finished.

        This function can be overridden in inherited classes to check other criteria for the readiness of the agent to
        execute main loop functions.
        """
        if not self.is_initialized or self.is_finalized:
            raise RuntimeError(f"The agent {self.id} has not been initialized and/or has already been finalized.")

    def execute_received_task(self, task: Task) -> Optional[Union[DataTensor, List[str]]]:
        """ Execute received method on the master.

        :param task: Received task to execute.
        :return: Execution result.
        """
        try:
            time_st = time.time()
            result = getattr(self, task.method_name)(**task.kwargs_dict)
            self.task_execution_times.append(
                TaskExecutionTime(
                    client_id=self.id,
                    task_name=str(task.method_name),
                    execution_time=time.time() - time_st
                )
            )
        except AttributeError as exc:
            raise UnsupportedError(f'Method {task.method_name} is not supported on {self.id}') from exc
        return result

    @abstractmethod
    def run(self, party: PartyCommunicator) -> None:
        """ Run the VFL model training with the party agent.

        Current method should implement initialization of the party agent, launching of the main training loop,
        and finalization of the experiment.

        :param party: Communicator instance used for communication between VFL agents.
        :return: None
        """
        ...

    @abstractmethod
    def loop(self, batcher: Batcher, party: PartyCommunicator) -> None:
        """ Run main training loop on the VFL agent.

        :param batcher: Batcher for creating training batches.
        :param party: Communicator instance used for communication between VFL agents.

        :return: None
        """
        ...

    @abstractmethod
    def inference_loop(self, batcher: Batcher, party: PartyCommunicator) -> None:
        """ Run main inference loop on the VFL agent.

        :param batcher: Batcher for creating inference batches.
        :param party: Communicator instance used for communication between VFL agents.

        :return: None
        """
        ...

    @abstractmethod
    def initialize(self, is_infer: bool = False):
        """ Initialize the party agent. """
        ...

    @abstractmethod
    def finalize(self, is_infer: bool = False):
        """ Finalize the party agent. """
        ...

    @abstractmethod
    def initialize_model_from_params(self, **model_params) -> Any:
        ...

    def save_model(self, is_ovr_models: bool = False):
        """ Save model for further inference. """
        if self._model is not None and self.do_save_model:
            if self.model_path is None:
                raise RuntimeError('If `do_save_model` is True, the `model_path` must be not None.')
            os.makedirs(os.path.join(self.model_path, f'agent_{self.id}'), exist_ok=True)
            logger.info(f"Agent {self.id} saving the model to {self.model_path}")
            if is_ovr_models:
                for idx, model in enumerate(self._model):
                    torch.save(model.state_dict(), os.path.join(self.model_path, f'agent_{self.id}', f'model-{idx}.pt'))
            else:
                torch.save(self._model.state_dict(), os.path.join(self.model_path, f'agent_{self.id}', 'model.pt'))
            with open(os.path.join(self.model_path, f'agent_{self.id}', 'model_init_params.json'), 'w') as f:
                if is_ovr_models:
                    init_params = {}
                    for idx, model in enumerate(self._model):
                        init_params[idx] = getattr(model, 'init_params', {})
                    json.dump(init_params, f)
                else:
                    json.dump(getattr(self._model, 'init_params', {}), f)

    def load_model(self, is_ovr_models: bool = False) -> Any:
        """ Load model saved for inference. """
        logger.info(f'{self.id} is loading model from {self.model_path}')
        if self.model_path is None:
            raise RuntimeError('If `do_load_model` is True, the `model_path` must be not None.')
        agent_model_path = os.path.join(self.model_path, f'agent_{self.id}')
        if not os.path.exists(agent_model_path):
            raise FileNotFoundError(
                f'You should train the model before launching the inference, however, {self.id} cannot find the model '
                f'at {os.path.join(self.model_path)}.'
            )

        with open(os.path.join(agent_model_path, 'model_init_params.json')) as f:
            init_model_params = json.load(f)

        if (model_params := getattr(self, '_model_params', None)) is not None:
            for k, v in model_params.items():
                init_model_params[k] = v

        if is_ovr_models:
            logger.info(f'Loading OVR models from {agent_model_path}')
            model = []
            for idx in sorted([int(key) for key in init_model_params.keys()]):
                m = self.initialize_model_from_params(**init_model_params[str(idx)])
                m.load_state_dict(torch.load(os.path.join(agent_model_path, f'model-{idx}.pt')))
                model.append(m)
        else:
            model = self.initialize_model_from_params(**init_model_params)
            model.load_state_dict(torch.load(os.path.join(self.model_path, f'agent_{self.id}', 'model.pt')))
        return model


class PartyMaster(PartyAgent, ABC):
    """ Abstract base class for the master party in the VFL experiment. """

    epochs: int
    report_train_metrics_iteration: int
    report_test_metrics_iteration: int
    target: DataTensor
    target_uids: List[str]
    test_target: DataTensor
    inference_target_uids: List[str]

    iteration_times: List[IterationTime] = list()

    @property
    def train_timings(self) -> list:
        """ Return list of tuples representing iteration timings from the main loop. """
        return self.iteration_times

    def synchronize_uids(
            self, collected_uids: list[Tuple[list[str], bool]], world_size: int, is_infer: bool = False
    ) -> List[str]:
        """ Synchronize unique records identifiers across party members.

        :param collected_uids: List of lists containing unique records identifiers collected from party members.
        :param world_size: Number of party members in the experiment.

        :return: Common records identifiers among the agents used in training loop.
        """
        logger.debug(f"Master {self.id}: synchronizing uuids for party of size {world_size}")
        inner_collected_uids = [col_uids[0] for col_uids in collected_uids if col_uids[1]]
        uids = self.inference_target_uids if is_infer else self.target_uids
        if len(inner_collected_uids) > 0:
            uids = itertools.chain(
                uids, (uid for member_uids in inner_collected_uids for uid in set(member_uids))
            )
        shared_uids = sorted(
            [uid for uid, count in collections.Counter(uids).items() if count == len(inner_collected_uids) + 1]
        )
        logger.debug(f"Master {self.id}: registering shared uuids of size {len(shared_uids)}")
        if is_infer:
            self.inference_target_uids = shared_uids
        else:
            self.target_uids = shared_uids
        logger.debug(f"Master {self.id}: record uuids has been successfully synchronized")
        return shared_uids

    @abstractmethod
    def report_metrics(self, y: DataTensor, predictions: DataTensor, name: str, step: int):
        """ Report metrics based on target values and predictions.

        :param y: Target values.
        :param predictions: Model predictions.
        :param name: Name of the dataset ("Train" or "Test").
        :param step: Iteration number.


        :return: None
        """
        ...


class PartyMember(PartyAgent, ABC):
    """ Abstract base class for the member party in the VFL experiment. """

    report_train_metrics_iteration: int
    report_test_metrics_iteration: int

    @abstractmethod
    def records_uids(self, is_infer: bool = False) -> Tuple[List[str], bool]:
        """ Get the list of existing dataset unique identifiers and either to use inner join for it

        :return: Tuple of unique identifiers and use_inner_join.
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


class UnsupportedError(Exception):
    """Custom exception class for indicating that an unsupported method is called on a class."""

    def __init__(self, message: str = "Unsupported method for class."):
        super().__init__(message)
