import asyncio
import logging
import time
import uuid
from collections import defaultdict
from concurrent import futures
from queue import Queue
from typing import Any, AsyncIterator, Optional, List

import grpc
from google.protobuf.message import Message

from stalactite.base import TaskExecutionTime, IterationTime
from stalactite.communications.grpc_utils.generated_code import (
    communicator_pb2,
    communicator_pb2_grpc,
)
from stalactite.communications.grpc_utils.utils import PrometheusMetric, Status

logger = logging.getLogger(__name__)
logging.getLogger('asyncio').setLevel(logging.ERROR)
logging.getLogger('grpc').setLevel(logging.ERROR)


class GRpcCommunicatorServicer(communicator_pb2_grpc.CommunicatorServicer):
    """
    gRPC asynchronous server and servicer class.

    Hold gRPC server endpoints, encapsulate logic on tasks` requests exchange between VFL members and master.
    """

    def __init__(
            self,
            world_size: int,
            master_id: str,
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
        self.port = port
        self.threadpool_max_workers = threadpool_max_workers
        self.max_message_size = max_message_size
        self.disconnect_idle_client_time = disconnect_idle_client_time
        self.run_prometheus = run_prometheus
        self.experiment_label = experiment_label
        self.time_between_idle_connections_checks = time_between_idle_connections_checks

        self.host = '0.0.0.0'

        if self.run_prometheus:
            if self.experiment_label is None:
                raise RuntimeError("Experiment label (`experiment_label`) is not set. Cannot log to Prometheus")
            self.prometheus_run_id = str(uuid.uuid4())
            logger.info(f'Created Prometheus run_id: {self.prometheus_run_id}')

        self.status = Status.not_started
        self.connected_clients = dict()

        self._received_tasks = defaultdict(lambda: defaultdict(Queue))
        self._tasks_to_send_queues = defaultdict(lambda: defaultdict(Queue))
        self._info_messages = 0

    def put_to_received_tasks(self, message: communicator_pb2.MainMessage, receive_from_id: str):
        self._received_tasks[message.method_name][receive_from_id].put(message)

    async def get_from_tasks_to_send_dict(
            self, method_name: str, send_to_id: str, timeout: float = 30.0
    ) -> communicator_pb2.MainMessage:
        timer_start = time.time()
        while (message_queue := self._tasks_to_send_queues.get(send_to_id, dict()).get(method_name, Queue())).empty():
            await asyncio.sleep(0.0)
            if time.time() - timer_start > timeout:
                raise TimeoutError(f"Could not send task: {method_name} to {send_to_id}.")
        message = message_queue.get()
        self.log_recv_message_size(message)
        return message

    @staticmethod
    def _message_size(message: Message) -> int:
        """
        Return protobuf message size in bytes

        :param message: Protobuf message
        """
        return message.ByteSize()

    def log_recv_message_size(self, message: communicator_pb2.MainMessage) -> None:
        """Log received by master message size in bytes to the Prometheus."""
        if self.run_prometheus:
            logger.debug("Reporting `task_message_size` to Prometheus.")
            PrometheusMetric.recv_message_size.value.labels(
                experiment_label=self.experiment_label,
                run_id=self.prometheus_run_id,
                task_type=message.method_name,
                client_id=message.sender_id,
            ).observe(self._message_size(message))

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
        server.add_insecure_port(f"{self.host}:{self.port}")
        logger.info(f"Starting server at {self.host}:{self.port}")
        asyncio.create_task(self._check_active_connections())
        await server.start()
        await server.wait_for_termination()

    def log_agents_status(self) -> None:
        """Log number of active clients to the Prometheus."""
        if self.run_prometheus:
            logger.debug("Reporting `number_of_connected_agents` to Prometheus.")
            PrometheusMetric.number_of_connected_agents.value.labels(
                experiment_label=self.experiment_label,
                run_id=self.prometheus_run_id
            ).set(len(self.connected_clients))

    def log_execution_time(self, execution_time: List[TaskExecutionTime]) -> None:
        """ Log execution time of the tasks on agents to Prometheus. """
        if self.run_prometheus and execution_time:
            logger.debug("Reporting `agent_task_execution_time` to Prometheus.")
            for exec_time in execution_time:
                PrometheusMetric.execution_time.value.labels(
                    experiment_label=self.experiment_label,
                    run_id=self.prometheus_run_id,
                    client_id=exec_time.client_id,
                    task_type=exec_time.task_name,
                ).observe(exec_time.execution_time)

    def log_iteration_time(self, iteration_time: List[IterationTime]) -> None:
        """ Log execution time of the tasks on agents to Prometheus. """
        if self.run_prometheus and iteration_time:
            logger.debug("Reporting `iteration_time_hist`, `iteration_time_gauge` to Prometheus.")
            for iter_time in iteration_time:
                PrometheusMetric.iteration_time_hist.value.labels(
                    experiment_label=self.experiment_label,
                    run_id=self.prometheus_run_id,
                ).observe(iter_time.iteration_time)
                PrometheusMetric.iteration_time_gauge.value.labels(
                    experiment_label=self.experiment_label,
                    run_id=self.prometheus_run_id,
                ).set(iter_time.iteration_time)

    def log_communication_time(self, client_id: str, timings: list[communicator_pb2.SendTime]) -> None:
        """
        Log time of the unary communication operations from members.

        :param client_id: ID of the member sent the metrics
        :param timings: List of the SendTime objects containing unary send time
        """
        if self.run_prometheus and timings:
            logger.debug("Reporting `member_send_time` to Prometheus.")
            for timing in timings:
                PrometheusMetric.send_client_time.value.labels(
                    experiment_label=self.experiment_label,
                    client_id=client_id,
                    task_type=timing.method_name,
                    run_id=self.prometheus_run_id,
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
                    self.log_agents_status()
                    self._info_messages = 0
                    logger.info(f"Client {conn_id} disconnected")

    def process_heartbeat(self, request: communicator_pb2.HB) -> communicator_pb2.HB:
        """
        Process heartbeats from members.
        Indicate whether all members ready should start.
        """
        client_name = request.agent_name
        logger.debug(f"Got ping from client {client_name}")

        cur_clients = list(self.connected_clients.keys())
        self.connected_clients[client_name] = time.time()
        if client_name not in cur_clients:
            self.log_agents_status()
        if len(self.connected_clients) == self.world_size:
            if not self._info_messages:
                logger.debug(f"All {self.world_size} clients connected")
                self._info_messages += 1
            self.status = Status.all_ready
        else:
            self.status = Status.waiting
        self.log_communication_time(client_id=client_name, timings=list(request.send_timings))
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
        self.log_recv_message_size(request)
        return communicator_pb2.MainMessage(
            sender_id=self.master_id,
        )

    async def RecvFromMaster(
            self, request: communicator_pb2.MainMessage, context: grpc.aio.ServicerContext
    ) -> communicator_pb2.MainMessage:
        get_task = asyncio.create_task(
            self.get_from_tasks_to_send_dict(
                method_name=request.method_name, send_to_id=request.sender_id, timeout=request.get_response_timeout
            )
        )
        result = await get_task
        return result
