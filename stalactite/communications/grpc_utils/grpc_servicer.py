import asyncio
import logging
import time
from collections import defaultdict
from concurrent import futures
from typing import Any, AsyncIterator, Optional

import grpc
from google.protobuf.message import Message

from stalactite.communications.grpc_utils.generated_code import (
    communicator_pb2,
    communicator_pb2_grpc,
)
from stalactite.communications.grpc_utils.utils import (
    PrometheusMetric,
    Status,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
sh = logging.StreamHandler()
sh.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
logger.addHandler(sh)


class GRpcCommunicatorServicer(communicator_pb2_grpc.CommunicatorServicer):
    """
    gRPC asynchronous server and servicer class.

    Hold gRPC server endpoints, encapsulate logic on tasks` requests exchange between VFL members and master.
    """

    def __init__(
        self,
        world_size: int,
        master_id: str,
        host: str,
        port: str,
        *args,
        threadpool_max_workers: int = 10,
        max_message_size: int = -1,
        logging_level: Any = logging.INFO,
        disconnect_idle_client_time: float = 120.0,
        run_prometheus: bool = False,
        experiment_label: Optional[str] = None,
        time_between_idle_connections_checks: float = 3.0,
        **kwargs,
    ) -> None:
        """
        Initialize GRpcCommunicatorServicer with necessary connection arguments.

        :param world_size: Number of VFL members (without the master), which will be connected
        :param master_id: Identifier of the VFL master
        :param host: Host of the gRPC server
        :param port: Port of the gRPC server
        :param args: Arguments of the communicator_pb2_grpc.CommunicatorServicer class
        :param threadpool_max_workers: Number of threadpool workers processing connections
        :param max_message_size: Maximum message length that the gRPC channel can send or receive. -1 means unlimited.
        :param disconnect_idle_client_time: Time in seconds to wait after a client`s last heartbeat to consider the
               client disconnected
        :param run_prometheus: Whether to report heartbeat metrics to Prometheus
        :param experiment_label: Prometheus metrics label for current experiment
        :param time_between_inactive_connections_checks: Time in sec between checking last client pings
        :param kwargs: Keyword arguments of the communicator_pb2_grpc.CommunicatorServicer class
        """
        super().__init__(*args, **kwargs)
        logger.setLevel(logging_level)
        self.world_size = world_size
        self.master_id = master_id
        self.host = host
        self.port = port
        self.threadpool_max_workers = threadpool_max_workers
        self.max_message_size = max_message_size
        self.disconnect_idle_client_time = disconnect_idle_client_time
        self.run_prometheus = run_prometheus
        self.experiment_label = experiment_label
        self.time_between_idle_connections_checks = time_between_idle_connections_checks

        if self.run_prometheus and self.experiment_label is None:
            raise RuntimeError("Experiment label (`experiment_label`) is not set. Cannot log heartbeats to Prometheus")

        self.status = Status.not_started
        self.connected_clients = dict()

        self._received_tasks = defaultdict(dict)
        self._tasks_to_send_queues = defaultdict(dict)

    def put_to_received_tasks(self, message: communicator_pb2.MainMessage, receive_from_id: str):
        self._received_tasks[message.method_name][receive_from_id] = message

    async def get_from_tasks_to_send_dict(
            self, method_name: str, send_to_id: str, timeout: float = 30.
    ) -> communicator_pb2.MainMessage:
        timer_start = time.time()
        while (message := self._tasks_to_send_queues.get(send_to_id, dict()).pop(method_name, None)) is None:
            await asyncio.sleep(0.)
            if time.time() - timer_start > timeout:
                raise TimeoutError(f'Could not send task: {method_name} to {send_to_id}.')
        return message

    @staticmethod
    def _message_size(message: Message) -> int:
        """
        Return protobuf message size in bytes

        :param message: Protobuf message
        """
        return message.ByteSize()

    def log_iteration_timings(self, iteration_times: list[tuple[int, float]]):
        """
        Log to the Prometheus info of the training iteration time on master.

        :param iteration_times: List of the iteration number and time pairs
        """
        if self.run_prometheus:
            logger.debug("Reporting `master_iteration_time` to Prometheus")
            for iter_num, iter_time in iteration_times:
                PrometheusMetric.iteration_times.value.labels(
                    experiment_label=self.experiment_label,
                    iteration=iter_num,
                ).observe(iter_time)

    def _log_agents_metrics(self, client_id: str, task_type: str, message_size: int, execution_time: float) -> None:
        """
        Log to the Prometheus info of the task execution time and result message size on clients.

        :param client_id: ID of the member sent the metrics
        :param task_type: Name of the task which was executed
        :param message_size: Size of the results message size in bytes
        :param execution_time: Task execution time
        """
        if self.run_prometheus:
            logger.debug("Reporting metrics to Prometheus")
            PrometheusMetric.message_size.value.labels(
                experiment_label=self.experiment_label,
                client_id=client_id,
                task_type=task_type,
            ).observe(message_size)
            PrometheusMetric.execution_time.value.labels(
                experiment_label=self.experiment_label,
                client_id=client_id,
                task_type=task_type,
            ).observe(execution_time)

    async def start_servicer_and_server(self):
        """Launch gRPC server and servicer."""
        server = grpc.aio.server(
            futures.ThreadPoolExecutor(max_workers=self.threadpool_max_workers),
            options=[
                ("grpc.max_send_message_length", self.max_message_size),
                ("grpc.max_receive_message_length", self.max_message_size),
            ],
        )
        communicator_pb2_grpc.add_CommunicatorServicer_to_server(self, server)
        server.add_insecure_port(f"{self.host}:{self.port}")  # TODO SSL goes here
        logger.info(f"Starting server at {self.host}:{self.port}")
        asyncio.create_task(self._check_active_connections())
        await server.start()
        await server.wait_for_termination()

    def _log_agents_status(self) -> None:
        """Log number of active clients to the Prometheus."""
        if self.run_prometheus:
            logger.debug("Reporting `number_of_connected_agents` to Prometheus.")
            PrometheusMetric.number_of_connected_agents.value.labels(experiment_label=self.experiment_label).set(
                len(self.connected_clients)
            )

    def _log_communication_time(self, client_id: str, timings: list[communicator_pb2.SendTime]) -> None:
        """
        Log time of the unary communication operations from members.

        :param client_id: ID of the member sent the metrics
        :param timings: List of the SendTime objects containing unary send time
        """
        if self.run_prometheus:
            logger.debug("Reporting `member_send_time` to Prometheus.")
            for timing in timings:
                PrometheusMetric.send_client_time.value.labels(
                    experiment_label=self.experiment_label,
                    client_id=client_id,
                    task_type=timing.method_name,
                ).observe(timing.send_time)

    async def _check_active_connections(self):
        """
        Monitor active connections.
        Remove clients which have not sent HB message in `disconnect_idle_client_time` sec.
        """
        while True:
            await asyncio.sleep(self.time_between_idle_connections_checks)
            items = list(self.connected_clients.items())
            for conn_id, last_ping in items:
                if time.time() - last_ping > self.disconnect_idle_client_time:
                    self.connected_clients.pop(conn_id)
                    self._log_agents_status()
                    logger.info(f"Client {conn_id} disconnected")

    def process_heartbeat(self, request: communicator_pb2.HB) -> communicator_pb2.HB:
        """
        Process heartbeats from members.
        Indicate whether all members ready should start.
        """
        client_name = request.agent_name
        logger.debug(f"Got ping from client {client_name}")
        self.connected_clients[client_name] = time.time()
        if len(self.connected_clients) == self.world_size:
            logger.debug(f"All {self.world_size} clients connected")
            self.status = Status.all_ready
        else:
            self.status = Status.waiting
        self._log_agents_status()
        self._log_communication_time(client_id=client_name, timings=list(request.send_timings))
        return communicator_pb2.HB(
            agent_name=self.master_id,
            status=self.status,
        )

    async def Heartbeat(
        self,
        request_iterator: AsyncIterator[communicator_pb2.HB],
        context: grpc.aio.ServicerContext,
    ) -> AsyncIterator[communicator_pb2.HB]:
        """Process heartbeats from member. Send world status back to member."""
        async for request in request_iterator:
            yield self.process_heartbeat(request)

    async def SendToMaster(
            self, request: communicator_pb2.MainMessage, context: grpc.aio.ServicerContext
    ) -> communicator_pb2.MainMessage:
        self.put_to_received_tasks(message=request, receive_from_id=request.sender_id)
        return communicator_pb2.MainMessage(
            sender_id=self.master_id,
        )

    async def RecvFromMaster(
            self, request: communicator_pb2.MainMessage, context: grpc.aio.ServicerContext
    ) -> communicator_pb2.MainMessage:
        get_task = asyncio.create_task(
            self.get_from_tasks_to_send_dict(
                method_name=request.method_name,
                send_to_id=request.sender_id,
                timeout=request.get_response_timeout
            )
        )
        result = await get_task
        return result
