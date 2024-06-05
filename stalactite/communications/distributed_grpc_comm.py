import asyncio
import logging
import threading
import time
import uuid
from abc import ABC
from collections import defaultdict
from queue import Queue
from typing import Any, Coroutine, Iterator, List, Optional, Union

import grpc
from prometheus_client import start_http_server

from stalactite.base import (
    ParticipantFuture,
    PartyCommunicator,
    PartyMaster,
    PartyMember,
    Task,
    UnsupportedError,
    PartyAgent,
)
from stalactite.communications.grpc_utils.generated_code import (
    communicator_pb2,
    communicator_pb2_grpc,
    arbitered_communicator_pb2,
    arbitered_communicator_pb2_grpc,
)
from stalactite.communications.grpc_utils.grpc_master_servicer import GRpcCommunicatorServicer
from stalactite.communications.grpc_utils.grpc_arbiter_servicer import GRpcArbiterCommunicatorServicer
from stalactite.communications.grpc_utils.utils import (
    ClientStatus,
    SerializedMethodMessage,
    Status,
    collect_kwargs,
    prepare_kwargs,
    start_thread,
)
from stalactite.communications.helpers import METHOD_VALUES, Method, MethodKwargs
from stalactite.ml.arbitered.base import PartyArbiter
from stalactite.utils import Role

logger = logging.getLogger(__name__)

PROMETHEUS_METRICS_PREFIX = "__prometheus_"


class GRpcPartyCommunicator(PartyCommunicator, ABC):
    """Base gRPC party communicator class."""

    participant: PartyAgent
    logging_level = logging.INFO
    is_ready: bool = False

    def __init__(
            self,
            logging_level: Any = logging.INFO,
            recv_timeout: float = 1200.0,
    ):
        """Initialize base GRpcPartyCommunicator class. Set the module logging level."""
        self.logging_level = logging_level
        self.recv_timeout = recv_timeout

        logger.setLevel(logging_level)

    def raise_if_not_ready(self):
        """Raise an exception if the communicator was not initialized properly."""
        if not self.is_ready:
            raise RuntimeError(
                "Cannot proceed because communicator is not ready. "
                "Perhaps, rendezvous has not been called or was unsuccessful"
            )


class GRpcMasterPartyCommunicator(GRpcPartyCommunicator):
    """gRPC Master communicator class.
    This class is used as the communicator for master in gRPC server-based (distributed) VFL setup.
    """

    MEMBER_DATA_FIELDNAME = "__member_data__"

    def __init__(
            self,
            participant: Union[PartyAgent, PartyMaster],
            world_size: int,
            port: Union[int, str],
            server_thread_pool_size: int = 10,
            max_message_size: int = -1,
            rendezvous_timeout: float = 3600.0,
            disconnect_idle_client_time: float = 120.0,
            prometheus_server_port: int = 8080,
            run_prometheus: bool = False,
            experiment_label: Optional[str] = None,
            time_between_idle_connections_checks: float = 3.0,
            use_arbiter: bool = False,
            arbiter_host: Optional[str] = None,
            arbiter_port: Optional[Union[int, str]] = False,
            sent_task_timout: float = 3600.,
            **kwargs,
    ):
        """
        Initialize master communicator with connection parameters.

        :param participant: PartyMaster instance
        :param world_size: Number of VFL member agents
        :param port: Port of the gRPC server
        :param server_thread_pool_size: Number of threadpool workers processing connections on the gRPC server
        :param max_message_size: Maximum message length that the gRPC channel can send or receive. -1 means unlimited
        :param rendezvous_timeout: Maximum time to wait until all the members are connected
        :param disconnect_idle_client_time: Time in seconds to wait after a client`s last heartbeat to consider the
               client disconnected
        :param prometheus_server_port: HTTP server on master started to report metrics to Prometheus
        :param run_prometheus: Whether to report heartbeat metrics to Prometheus
        :param experiment_label: Label of the experiment used in prerequisites
        :param time_between_idle_connections_checks: Time in sec between checking last client pings
        """
        self.participant = participant
        self.world_size = world_size
        self.port = port
        self.server_thread_pool_size = server_thread_pool_size
        self.max_message_size = max_message_size
        self.rendezvous_timeout = rendezvous_timeout
        self.disconnect_idle_client_time = disconnect_idle_client_time
        self.run_prometheus = run_prometheus
        self.prometheus_server_port = prometheus_server_port
        self.experiment_label = experiment_label
        self.time_between_idle_connections_checks = time_between_idle_connections_checks
        self.use_arbiter = use_arbiter
        self.arbiter_port = arbiter_port
        self.arbiter_host = arbiter_host
        self.sent_task_timout = sent_task_timout

        self.server_thread = None
        self.asyncio_event_loop = None
        self.servicer: Optional[GRpcCommunicatorServicer] = None

        self._arbiter_stub = None
        self._grpc_channel_arbiter = None

        self.arbiter_ready: bool = False
        self.arbiter: Optional[str] = None

        if self.use_arbiter:
            if self.arbiter_port is None or self.arbiter_host is None:
                raise ValueError('If `use_arbiter` is True, `arbiter_host` and `arbiter_port` must be not None.')

        super().__init__(**kwargs)

    @property
    def servicer_initialized(self) -> bool:
        """Whether the gRPC server was started and CommunicatorServicer was initialized."""
        return self.servicer is not None

    @property
    def status(self) -> Status:
        """Return status of the master communicator."""
        if not self.servicer_initialized:
            return Status.not_started
        return self.servicer.status

    @property
    def number_connected_clients(self) -> int:
        """Return number of VFL agent members connected to the server."""
        if not self.servicer_initialized:
            return 0
        return len(self.servicer.connected_clients)

    @property
    def members(self) -> list[str]:
        """List the VFL agent members` ids connected to the server."""
        return list(self.servicer.connected_clients.keys())

    def rendezvous(self) -> None:
        """Wait until all the VFL agent members are connected to the gRPC server."""
        timer = time.time()
        if not self.servicer_initialized:
            raise ValueError("Started rendezvous before initializing gRPC server and servicer")
        if self.use_arbiter:
            not_ready_condition = lambda: self.status != Status.all_ready or not self.arbiter_ready
        else:
            not_ready_condition = lambda: self.status != Status.all_ready
        while not_ready_condition():
            if self.use_arbiter and not self.arbiter_ready:
                try:
                    arbiter_msg = self._arbiter_stub.CheckIfAvailable(
                        arbitered_communicator_pb2.IsReady(
                            sender_id=self.participant.id,
                            ready=True,
                            role=Role.master,
                        ),
                        timeout=self.sent_task_timout,
                    )
                    self.arbiter_ready = arbiter_msg.ready
                    self.arbiter = arbiter_msg.sender_id
                except:
                    continue
            time.sleep(0.1)
            if time.time() - timer > self.rendezvous_timeout:
                raise TimeoutError(
                    "Rendezvous timeout. You can try to set larger value in `rendezvous_timeout` param"
                )
        if self.number_connected_clients != self.world_size:
            raise RuntimeError("Rendezvous failed")

    @property
    def is_ready(self) -> bool:
        """Return True if the gRPC server is ready to accept connections and process the requests."""
        if self.use_arbiter:
            return self.server_thread.is_alive() and self.arbiter_ready
        return self.server_thread.is_alive()

    def put_to_tasks_to_send_queue(self, message: communicator_pb2.MainMessage, send_to_id: str):
        self.servicer._tasks_to_send_queues[send_to_id][message.method_name] = message

    def send(
            self,
            send_to_id: str,
            method_name: Method,
            method_kwargs: Optional[MethodKwargs] = None,
            result: Optional[Any] = None,
            **kwargs,
    ) -> Task:
        """
        Send task to VFL member via gRPC channel.

        :param send_to_id: Identifier of the VFL member receiver
        :param method_name: Method name to execute on participant
        :param require_answer: True if the task requires answer to sent back to sender
        :param task_id: Unique identifier of the task
        :param parent_id: Unique identifier of the parent task
        :param parent_method_name: Method name of the parent task
        :param parent_task_execution_time: Time in seconds of the parent task execution time
        :param message: Task data arguments
        :param kwargs: Optional kwargs which are ignored
        """
        self.raise_if_not_ready()

        if kwargs:
            logger.warning(f"Got unexpected kwargs in PartyCommunicator.sent method {kwargs}. Omitting.")

        if result is not None:
            attr = METHOD_VALUES.get(method_name, "other_kwargs")
            if method_kwargs is None:
                method_kwargs = MethodKwargs()
            kwargs = getattr(method_kwargs, attr)
            kwargs["result"] = result
            setattr(method_kwargs, attr, kwargs)

        message_kwargs = prepare_kwargs(method_kwargs)
        task_id = str(uuid.uuid4())

        if self.arbiter == send_to_id:
            prepared_task_message = arbitered_communicator_pb2.MainArbiterMessage(
                sender_id=self.participant.id,
                task_id=task_id,
                method_name=method_name,
                tensor_kwargs=message_kwargs.tensor_kwargs,
                other_kwargs=message_kwargs.other_kwargs,
            )
        else:
            prepared_task_message = communicator_pb2.MainMessage(
                sender_id=self.participant.id,
                task_id=task_id,
                method_name=method_name,
                tensor_kwargs=message_kwargs.tensor_kwargs,
                other_kwargs=message_kwargs.other_kwargs,
                prometheus_metrics=message_kwargs.prometheus_kwargs,
            )

        if self.participant.id == send_to_id:
            self.servicer.put_to_received_tasks(prepared_task_message, receive_from_id=self.participant.id)
        elif self.arbiter == send_to_id:
            res = self._arbiter_stub.SendToArbiter(prepared_task_message, timeout=self.sent_task_timout)
            assert res.sender_id == self.arbiter, "Sent message was not acknowledged"
        else:
            self.put_to_tasks_to_send_queue(prepared_task_message, send_to_id=send_to_id)

        logger.debug("Party communicator %s: sent to %s event %s" % (self.participant.id, send_to_id, task_id))
        return Task(id=task_id, method_name=method_name, to_id=send_to_id, from_id=self.participant.id)

    def _get_all_agents(self):
        if self.use_arbiter:
            logger.warning('Defaulting to all the agents (members and arbiter)')
            return self.members + [self.arbiter]
        logger.warning('Defaulting to all the members (without arbiter)')
        return self.members

    def scatter(
            self,
            method_name: Method,
            method_kwargs: Optional[List[MethodKwargs]] = None,
            result: Optional[Union[Any, List[Any]]] = None,
            participating_members: Optional[List[str]] = None,
            **kwargs,
    ) -> List[Task]:

        if participating_members is None:
            participating_members = self._get_all_agents()

        if method_kwargs is not None:
            assert len(method_kwargs) == len(participating_members), (
                f"Number of tasks in scatter operation ({len(method_kwargs)}) is not equal to the "
                f"`participating_members` number ({len(participating_members)})"
            )
        else:
            method_kwargs = [None for _ in range(len(participating_members))]

        if isinstance(result, list):
            assert len(result) == len(participating_members), (
                f"Number of results in scatter operation ({len(result)}) is not equal to the "
                f"`participating_members` number ({len(participating_members)})"
            )
        else:
            result = [result for _ in range(len(participating_members))]

        tasks = []
        for send_to_id, m_kwargs, res in zip(participating_members, method_kwargs, result):
            tasks.append(
                self.send(
                    send_to_id=send_to_id,
                    method_name=method_name,
                    method_kwargs=m_kwargs,
                    result=res,
                    **kwargs,
                )
            )
        return tasks

    def broadcast(
            self,
            method_name: Method,
            method_kwargs: Optional[MethodKwargs] = None,
            result: Optional[Any] = None,
            participating_members: Optional[List[str]] = None,
            include_current_participant: bool = False,
            **kwargs,
    ) -> List[Task]:

        if participating_members is None:
            participating_members = self._get_all_agents()

        if self.participant.id not in participating_members and include_current_participant:
            participating_members.append(self.participant.id)

        if method_kwargs is not None and not isinstance(method_kwargs, MethodKwargs):
            raise TypeError(
                "communicator.broadcast `method_kwargs` must be either None or MethodKwargs. "
                f"Got {type(method_kwargs)}"
            )

        tasks = []
        for send_to_id in participating_members:
            tasks.append(
                self.send(
                    send_to_id=send_to_id,
                    method_name=method_name,
                    method_kwargs=method_kwargs,
                    result=result,
                    **kwargs,
                )
            )
        return tasks

    def get_from_received_tasks(
            self,
            method_name: str,
            receive_from_id: str,
            timeout: float = 30.0,
            task_id: Optional[str] = None,
    ) -> communicator_pb2.MainMessage:
        timer_start = time.time()
        if self.use_arbiter:
            if receive_from_id == self.arbiter:
                request_message = communicator_pb2.MainMessage(
                    sender_id=self.participant.id,
                    task_id=task_id,
                    method_name=method_name,
                    get_response_timeout=timeout,
                )
                return self._arbiter_stub.RecvFromArbiter(request_message, timeout=timeout)
        while (message := self.servicer._received_tasks.get(method_name, dict()).pop(receive_from_id, None)) is None:
            if time.time() - timer_start > timeout:
                raise TimeoutError(f"Could not receive task: {method_name} from {receive_from_id}.")
            continue
        return message

    def recv(self, task: Task, recv_results: bool = False) -> Task:
        received_message = self.get_from_received_tasks(
            method_name=task.method_name,
            receive_from_id=task.to_id if recv_results else task.from_id,
            timeout=self.recv_timeout,
            task_id=task.id,
        )

        if task.from_id != received_message.sender_id and not recv_results:
            raise RuntimeError(
                f"Received task.from_id ({task.from_id}) differs from " f"`sender_id` ({received_message.sender_id})"
            )
        elif task.to_id != received_message.sender_id and recv_results:
            raise RuntimeError(
                f"Received task.to_id ({task.to_id}) differs from `sender_id` ({received_message.sender_id}). "
                "Check `recv_results` kwarg in recv / gather operation."
            )

        _prometheus_metrics = getattr(received_message, 'prometheus_metrics', None)

        received_kwargs, prometheus_metrics, result = collect_kwargs(
            SerializedMethodMessage(
                tensor_kwargs=dict(received_message.tensor_kwargs),
                other_kwargs=dict(received_message.other_kwargs),
            ),
            prometheus_metrics=_prometheus_metrics,
        )

        if prometheus_metrics:
            self.servicer.log_execution_time(prometheus_metrics[PROMETHEUS_METRICS_PREFIX + 'execution_time'])

        return Task(
            method_name=received_message.method_name,
            from_id=received_message.sender_id,
            to_id=self.participant.id,
            id=received_message.task_id,
            method_kwargs=received_kwargs,
            result=result,
        )

    def gather(self, tasks: List[Task], recv_results: bool = False) -> List[Task]:
        return [self.recv(task, recv_results=recv_results) for task in tasks]

    def run_coroutine(self, coroutine: Coroutine):
        """
        Run coroutine in the created asyncio event loop.
        Used to launch asyncio gRPC server in a separate thread.
        """
        self.asyncio_event_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.asyncio_event_loop)
        self.asyncio_event_loop.run_until_complete(coroutine)

    def run(self):
        """
        Run the VFL master.
        Launch the gRPC server, wait until all the members are connected and start sending, receiving and processing
        tasks from the main loop.
        """
        try:
            if self.run_prometheus:
                logger.info(f"Prometheus experiment label: {self.experiment_label}")
                start_http_server(port=self.prometheus_server_port)
            self.servicer = GRpcCommunicatorServicer(
                world_size=self.world_size,
                master_id=self.participant.id,
                port=self.port,
                threadpool_max_workers=self.server_thread_pool_size,
                max_message_size=self.max_message_size,
                logging_level=self.logging_level,
                disconnect_idle_client_time=self.disconnect_idle_client_time,
                run_prometheus=self.run_prometheus,
                experiment_label=self.experiment_label,
                time_between_idle_connections_checks=self.time_between_idle_connections_checks,
            )
            if self.use_arbiter:
                self._grpc_channel_arbiter = grpc.insecure_channel(
                    f"{self.arbiter_host}:{self.arbiter_port}",
                    options=[
                        ("grpc.max_send_message_length", self.max_message_size),
                        ("grpc.max_receive_message_length", self.max_message_size),
                    ],
                )
                self._arbiter_stub = arbitered_communicator_pb2_grpc.ArbiteredCommunicatorStub(
                    self._grpc_channel_arbiter)
                logger.info(f"Connecting to the arbiter at {self.arbiter_host}:{self.arbiter_port}")

            with start_thread(
                    target=self.run_coroutine,
                    args=(self.servicer.start_servicer_and_server(),),
                    daemon=True,
                    thread_timeout=2.0,
            ) as self.server_thread:
                self.rendezvous()
                self.master = self.participant.id
                self.participant.members = self.members
                self.participant.master = self.participant.id
                self.participant.arbiter = self.arbiter

                with start_thread(
                        target=self.log_master_metrics,
                        daemon=True,
                        thread_timeout=5.0
                ):
                    self.participant.run(self)

            logger.info("Party communicator %s: finished" % self.participant.id)

        except:
            logger.error("Exception in party communicator %s" % self.participant.id, exc_info=True)
            raise

    def log_master_metrics(self):
        while not self.participant.is_finalized:
            exec_timings = self.participant.task_execution_times
            self.participant.task_execution_times = list()

            iter_timings = self.participant.iteration_times
            self.participant.iteration_times = list()

            self.servicer.log_execution_time(exec_timings)
            self.servicer.log_iteration_time(iter_timings)
            time.sleep(3.)


class GRpcMemberPartyCommunicator(GRpcPartyCommunicator):
    """gRPC Member communicator class.
    This class is used as the communicator for member in gRPC server-based (distributed) VFL setup.
    """

    MEMBER_DATA_FIELDNAME = "__member_data__"

    def __init__(
            self,
            participant: PartyMember,
            master_host: str,
            master_port: Union[int, str],
            max_message_size: int = -1,
            heartbeat_interval: float = 2.0,
            sent_task_timout: float = 3600.0,
            rendezvous_timeout: float = 3600.0,
            use_arbiter: bool = False,
            arbiter_host: Optional[str] = None,
            arbiter_port: Optional[Union[int, str]] = None,
            **kwargs,
    ):
        """
        Initialize member communicator with connection and task parameters.

        :param participant: PartyMember instance
        :param master_port: Port of the gRPC server
        :param master_host: Host of the gRPC server
        :param max_message_size: Maximum message length that the gRPC channel can send or receive. -1 means unlimited
        :param heartbeat_interval: Interval in seconds between heartbeats from member to master
        :param sent_task_timout: Time to wait before task is marked `acknowledged` by the master
        :param rendezvous_timeout: Maximum time to wait until all the members are connected
        """
        self.participant = participant
        self.master_host = master_host
        self.master_port = master_port
        self.max_message_size = max_message_size
        self.heartbeat_interval = heartbeat_interval
        self.sent_task_timout = sent_task_timout
        self.rendezvous_timeout = rendezvous_timeout
        self.use_arbiter = use_arbiter
        self.arbiter_host = arbiter_host
        self.arbiter_port = arbiter_port

        if self.use_arbiter:
            if self.arbiter_port is None or self.arbiter_host is None:
                raise ValueError('If `use_arbiter` is True, `arbiter_host` and `arbiter_port` must be not None.')

        self.master = None
        self._world_status = None
        self._heartbeats_thread = None

        self.tasks_futures: dict[str, ParticipantFuture] = dict()
        self._sent_time: Queue = Queue()
        self._lock = threading.Lock()

        self._received_tasks = defaultdict(dict)

        self.arbiter_ready: bool = False
        self.arbiter: Optional[str] = None

        super().__init__(**kwargs)

    def _start_client(self):
        """Create a gRPC channel. Start a thread with heartbeats from the member."""
        logger.info(f'Connecting to master at {self.master_host}:{self.master_port}')
        self._grpc_channel = grpc.insecure_channel(
            f"{self.master_host}:{self.master_port}",
            options=[
                ("grpc.max_send_message_length", self.max_message_size),
                ("grpc.max_receive_message_length", self.max_message_size),
            ],
        )
        self._stub = communicator_pb2_grpc.CommunicatorStub(self._grpc_channel)
        if self.use_arbiter:
            logger.info(f'Connecting to arbiter at {self.arbiter_host}:{self.arbiter_port}')
            self._grpc_arbiter_channel = grpc.insecure_channel(
                f"{self.arbiter_host}:{self.arbiter_port}",
                options=[
                    ("grpc.max_send_message_length", self.max_message_size),
                    ("grpc.max_receive_message_length", self.max_message_size),
                ],
            )
            self._arbiter_stub = arbitered_communicator_pb2_grpc.ArbiteredCommunicatorStub(self._grpc_arbiter_channel)

        logger.info(f"Starting ping-pong with the server {self.master_host}:{self.master_port}")
        pingpong_responses = self._stub.Heartbeat(self._heartbeats(), wait_for_ready=True)
        self._heartbeats_thread = threading.Thread(
            name="heartbeat-thread",
            target=self._read_server_heartbeats,
            args=(pingpong_responses,),
            daemon=True,
        )
        self._heartbeats_thread.start()

    def _get_all_from_sent_time_queue(self) -> list[Any]:
        """Reset the self._sent_time queue, returning all the elements from it."""
        with self._lock:
            elements = self._sent_time.queue
            self._sent_time = Queue()
            return list(elements)

    def _heartbeats(self):
        """Generate heartbeats messages."""
        while True:
            time.sleep(self.heartbeat_interval)
            heartbeat_message = communicator_pb2.HB(agent_name=self.participant.id, status=ClientStatus.alive)
            if len(timings := self._get_all_from_sent_time_queue()) > 0:
                heartbeat_message.send_timings.extend(timings)
            yield heartbeat_message

    def _read_server_heartbeats(self, server_responses: Iterator[communicator_pb2.HB]):
        """
        Read responses to heartbeats from master.
        Update info on the world status.

        :param server_responses: Iterator of the server responses
        """
        for response in server_responses:
            logger.debug(f"Got pong from master: {response.agent_name}")
            self._world_status = response.status
            if self.master is None:
                self.master = response.agent_name
            else:
                assert (
                        self.master == response.agent_name
                ), "Unexpected behaviour: Master id changed during the experiment"

    def rendezvous(self) -> None:
        """Wait until VFL master identify readiness to start of all the VFL members."""
        timer = time.time()
        if self.use_arbiter:
            not_ready_condition = lambda: self._world_status != Status.all_ready or not self.arbiter_ready
        else:
            not_ready_condition = lambda: self._world_status != Status.all_ready
        while not_ready_condition():
            if self.use_arbiter:
                if not self.arbiter_ready:
                    try:
                        arbiter_msg = self._arbiter_stub.CheckIfAvailable(
                            arbitered_communicator_pb2.IsReady(sender_id=self.participant.id, ready=True,
                                                               role=Role.member),
                            timeout=self.sent_task_timout,
                        )
                        self.arbiter_ready = arbiter_msg.ready
                        self.arbiter = arbiter_msg.sender_id
                    except:
                        continue
            time.sleep(0.1)
            if time.time() - timer > self.rendezvous_timeout:
                raise TimeoutError("Rendezvous timeout. You can try to set larger value in `rendezvous_timeout` param")
        logger.info(f"Client {self.participant.id} is ready to run")

    @property
    def is_ready(self) -> bool:
        """Return True if the VFL master is found and other members are alive."""
        if self.use_arbiter:
            return (self.master is not None) and (self._world_status == Status.all_ready) and self.arbiter_ready
        return (self.master is not None) and (self._world_status == Status.all_ready)

    def send(
            self,
            send_to_id: str,
            method_name: Method,
            method_kwargs: Optional[MethodKwargs] = None,
            result: Optional[Any] = None,
            **kwargs,
    ) -> Task:
        """
        Send task to VFL master via gRPC channel.

        :param send_to_id: Identifier of the VFL receiver (only master is available for member`s send
        :param method_name: Method name to execute on participant
        :param require_answer: True if the task requires answer to sent back to sender
        :param task_id: Unique identifier of the task
        :param parent_id: Unique identifier of the parent task
        :param parent_method_name: Method name of the parent task
        :param parent_task_execution_time: Time in seconds of the parent task execution time
        :param message: Task data arguments
        :param kwargs: Optional kwargs which are ignored
        """

        self.raise_if_not_ready()

        if send_to_id not in (self.master, self.participant.id, self.arbiter):
            raise UnsupportedError("GRpcMemberPartyCommunicator cannot send to other Members")

        prometheus_metrics = dict()
        logging_execution_timings = self.participant.task_execution_times
        self.participant.task_execution_times = list()
        prometheus_metrics[PROMETHEUS_METRICS_PREFIX + 'execution_time'] = logging_execution_timings

        if kwargs:
            logger.warning(f"Got unexpected kwargs in PartyCommunicator.sent method {kwargs}. Omitting.")

        if result is not None:
            attr = METHOD_VALUES.get(method_name, "other_kwargs")
            if method_kwargs is None:
                method_kwargs = MethodKwargs()
            kwargs = getattr(method_kwargs, attr)
            kwargs["result"] = result
            setattr(method_kwargs, attr, kwargs)

        message_kwargs = prepare_kwargs(method_kwargs, prometheus_metrics=prometheus_metrics)
        task_id = str(uuid.uuid4())

        if send_to_id == self.master:
            prepared_task_message = communicator_pb2.MainMessage(
                sender_id=self.participant.id,
                task_id=task_id,
                method_name=method_name,
                tensor_kwargs=message_kwargs.tensor_kwargs,
                other_kwargs=message_kwargs.other_kwargs,
                prometheus_metrics=message_kwargs.prometheus_kwargs,
            )

        else:
            prepared_task_message = arbitered_communicator_pb2.MainArbiterMessage(
                sender_id=self.participant.id,
                task_id=task_id,
                method_name=method_name,
                tensor_kwargs=message_kwargs.tensor_kwargs,
                other_kwargs=message_kwargs.other_kwargs,
            )

        if send_to_id in (self.master, self.arbiter):
            start = time.time()
            if send_to_id == self.master:
                res = self._stub.SendToMaster(prepared_task_message, timeout=self.sent_task_timout)
                assert res.sender_id == self.master, "Sent message was not acknowledged"
            elif send_to_id == self.arbiter and self.use_arbiter:
                res = self._arbiter_stub.SendToArbiter(prepared_task_message, timeout=self.sent_task_timout)
                assert res.sender_id == self.arbiter, "Sent message was not acknowledged"
            sent_timing = time.time() - start
            self._sent_time.put(
                communicator_pb2.SendTime(
                    task_id=task_id,
                    method_name=method_name,
                    send_time=sent_timing,
                )
            )
        else:
            self._received_tasks[prepared_task_message.method_name][self.participant.id] = prepared_task_message

        logger.debug("Party communicator %s: sent to %s event %s" % (self.participant.id, send_to_id, task_id))
        return Task(id=task_id, method_name=method_name, to_id=send_to_id, from_id=self.participant.id)

    def scatter(
            self,
            method_name: Method,
            method_kwargs: Optional[List[MethodKwargs]] = None,
            result: Optional[Union[Any, List[Any]]] = None,
            participating_members: Optional[List[str]] = None,
            **kwargs,
    ) -> List[Task]:
        raise UnsupportedError("GRpcMemberPartyCommunicator cannot scatter to other Members")

    def broadcast(
            self,
            method_name: Method,
            method_kwargs: Optional[MethodKwargs] = None,
            result: Optional[Any] = None,
            participating_members: Optional[List[str]] = None,
            include_current_participant: bool = False,
            **kwargs,
    ) -> List[Task]:
        """
        Broadcast task to VFL agents via gRPC channel.
        This method is unavailable for GRpcMemberPartyCommunicator as it cannot communcate with other members.
        """
        raise UnsupportedError("GRpcMemberPartyCommunicator cannot broadcast to other Members")

    def recv(self, task: Task, recv_results: bool = False) -> Task:
        receive_from_id = task.to_id if recv_results else task.from_id
        receive_to_id = task.from_id if recv_results else task.to_id

        if receive_from_id not in (self.master, self.participant.id, self.arbiter):
            raise UnsupportedError("GRpcMemberPartyCommunicator cannot receive from other Members")

        if receive_from_id == self.participant.id:
            received_message = self._received_tasks.get(task.method_name, dict()).pop(self.participant.id, None)
            if received_message is None:
                raise RuntimeError(f"Tried to receive task ({task.method_name}) from self before sending it to self.")
        else:
            if self.participant.id != receive_to_id:
                raise RuntimeError(
                    f"Tried to receive task for another participant: self.id {self.participant.id}; "
                    f"Task to_id: {receive_to_id}"
                )
            if receive_from_id == self.master:
                stub_call = self._stub.RecvFromMaster
                msg_type = communicator_pb2.MainMessage
            elif receive_from_id == self.arbiter and self.use_arbiter:
                stub_call = self._arbiter_stub.RecvFromArbiter
                msg_type = arbitered_communicator_pb2.MainArbiterMessage
            else:
                raise RuntimeError('Check the sender of the task and if the `use_arbiter` is set correctly')
            request_message = msg_type(
                sender_id=self.participant.id,
                task_id=task.id,
                method_name=task.method_name,
                get_response_timeout=self.recv_timeout,
            )
            received_message = stub_call(request_message, timeout=self.recv_timeout)

        if receive_from_id != received_message.sender_id:
            raise RuntimeError(
                f"Received task.from_id ({receive_from_id}) differs from " f"`sender_id` ({received_message.sender_id})"
            )
        _prometh_metrics = dict(received_message.prometheus_metrics) if not self.use_arbiter else None

        received_kwargs, prometheus_metrics, result = collect_kwargs(
            SerializedMethodMessage(
                tensor_kwargs=dict(received_message.tensor_kwargs),
                other_kwargs=dict(received_message.other_kwargs),
            ),
            prometheus_metrics=_prometh_metrics,
        )

        if prometheus_metrics:
            logger.warning("Got `prometheus_metrics` from master to member. This is not supposed to happen.")

        return Task(
            method_name=received_message.method_name,
            from_id=received_message.sender_id,
            to_id=self.participant.id,
            id=received_message.task_id,
            method_kwargs=received_kwargs,
            result=result,
        )

    def gather(self, tasks: List[Task], recv_results: bool = False) -> List[Task]:
        raise UnsupportedError("GRpcMemberPartyCommunicator cannot gather from other Members")

    def run(self):
        """
        Run the VFL member.
        Start the gRPC client threads, wait until server sends an `all ready` heartbeat response. Start requesting,
        receiving and processing tasks from the VFL master.
        """
        try:
            logger.info(f"Starting communicator {self.participant.id}")
            self._start_client()
            self.rendezvous()
            self.participant.members = None
            self.participant.master = self.master
            self.participant.arbiter = self.arbiter
            self.participant.run(self)
            while len(self._sent_time.queue) > 0:
                continue
            logger.info(f"Party communicator {self.participant.id} finished")
        except:
            logger.error("Exception in party communicator %s" % self.participant.id, exc_info=True)
            raise


class GRpcArbiterPartyCommunicator(GRpcMasterPartyCommunicator):
    def __init__(
            self,
            participant: PartyArbiter,
            world_size: int,
            port: Union[int, str],
            server_thread_pool_size: int = 10,
            max_message_size: int = -1,
            rendezvous_timeout: float = 3600.,
            **kwargs,
    ):
        """
        Initialize arbiter communicator with connection parameters.

        :param participant: PartyArbiter instance
        :param world_size: Number of members in the VFL setting (excluding master and arbiter)
        :param port: Port of the gRPC server
        :param server_thread_pool_size: Number of threadpool workers processing connections on the gRPC server
        :param max_message_size: Maximum message length that the gRPC channel can send or receive. -1 means unlimited
        """
        self.participant = participant
        self.world_size = world_size
        self.port = port
        self.server_thread_pool_size = server_thread_pool_size
        self.max_message_size = max_message_size
        self.rendezvous_timeout = rendezvous_timeout

        self.server_thread = None
        self.asyncio_event_loop = None
        self.servicer: Optional[GRpcArbiterCommunicatorServicer] = None

        super().__init__(
            participant=participant,
            world_size=world_size,
            port=port,
            server_thread_pool_size=server_thread_pool_size,
            max_message_size=max_message_size,
            rendezvous_timeout=rendezvous_timeout,
            **kwargs
        )

    @property
    def servicer_initialized(self) -> bool:
        """Whether the gRPC server was started and CommunicatorServicer was initialized."""
        return self.servicer is not None

    @property
    def status(self) -> Status:
        """Return status of the master communicator."""
        if not self.servicer_initialized:
            return Status.not_started
        return self.servicer.status

    def rendezvous(self) -> None:
        timer = time.time()
        if not self.servicer_initialized:
            raise ValueError("Started rendezvous before initializing gRPC server and servicer")
        while self.status != Status.all_ready:
            time.sleep(0.1)
            if time.time() - timer > self.rendezvous_timeout:
                raise TimeoutError(
                    "Rendezvous timeout. You can try to set larger value in `rendezvous_timeout` param"
                )

    @property
    def agents(self) -> list[str]:
        """List the VFL agent members` ids connected to the server."""
        return list(self.servicer.connected_agents.keys())

    def _get_all_agents(self):
        logger.warning('Defaulting to all the members (without arbiter)')
        return self.agents

    def recv(self, task: Task, recv_results: bool = False) -> Task:
        received_message = self.get_from_received_tasks(
            method_name=task.method_name,
            receive_from_id=task.to_id if recv_results else task.from_id,
            timeout=self.recv_timeout,
            task_id=task.id,
        )

        if task.from_id != received_message.sender_id and not recv_results:
            raise RuntimeError(
                f"Received task.from_id ({task.from_id}) differs from " f"`sender_id` ({received_message.sender_id})"
            )
        elif task.to_id != received_message.sender_id and recv_results:
            raise RuntimeError(
                f"Received task.to_id ({task.to_id}) differs from `sender_id` ({received_message.sender_id}). "
                "Check `recv_results` kwarg in recv / gather operation."
            )

        received_kwargs, prometheus_metrics, result = collect_kwargs(
            SerializedMethodMessage(
                tensor_kwargs=dict(received_message.tensor_kwargs),
                other_kwargs=dict(received_message.other_kwargs),
            ),
            prometheus_metrics=None,
        )
        return Task(
            method_name=received_message.method_name,
            from_id=received_message.sender_id,
            to_id=self.participant.id,
            id=received_message.task_id,
            method_kwargs=received_kwargs,
            result=result,
        )

    @property
    def members(self) -> list[str]:
        """List the VFL agent members` ids connected to the server."""
        return list(self.servicer.members)

    @property
    def master(self) -> Optional[str]:
        """List the VFL agent members` ids connected to the server."""
        return self.servicer.master

    def run(self):
        """
        Run the VFL arbiter.
        Launch the gRPC server, wait until all the agents are connected and start sending, receiving and processing
        tasks from the main loop.
        """
        try:
            self.servicer = GRpcArbiterCommunicatorServicer(
                world_size=self.world_size,
                arbiter_id=self.participant.id,
                port=self.port,
                threadpool_max_workers=self.server_thread_pool_size,
                max_message_size=self.max_message_size,
            )
            with start_thread(
                    target=self.run_coroutine,
                    args=(self.servicer.start_servicer_and_server(),),
                    daemon=True,
                    thread_timeout=2.0,
            ) as self.server_thread:
                self.rendezvous()
                self.participant.members = self.members
                self.participant.master = self.master
                self.participant.run(self)
            logger.info("Party communicator %s: finished" % self.participant.id)
        except:
            logger.error("Exception in party communicator %s" % self.participant.id, exc_info=True)
            raise
