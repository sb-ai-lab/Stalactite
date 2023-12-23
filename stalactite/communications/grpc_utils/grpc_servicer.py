import asyncio
from collections import defaultdict
import logging
from typing import AsyncIterator
from concurrent import futures

import grpc
from stalactite.communications.grpc_utils.generated_code import communicator_pb2, communicator_pb2_grpc
from stalactite.communications.grpc_utils.utils import Status, MessageTypes

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
sh = logging.StreamHandler()
sh.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
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
            **kwargs) -> None:
        """
        Initialize GRpcCommunicatorServicer with necessary connection arguments.

        :param world_size: Number of VFL members (without the master), which will be connected
        :param master_id: Identifier of the VFL master
        :param host: Host of the gRPC server
        :param port: Port of the gRPC server
        :param args: Arguments of the communicator_pb2_grpc.CommunicatorServicer class
        :param threadpool_max_workers: Number of threadpool workers processing connections
        :param max_message_size: Maximum message length that the gRPC channel can send or receive. -1 means unlimited.
        :param kwargs: Keyword arguments of the communicator_pb2_grpc.CommunicatorServicer class
        """
        super().__init__(*args, **kwargs)
        self.world_size = world_size
        self.master_id = master_id
        self.host = host
        self.port = port
        self.threadpool_max_workers = threadpool_max_workers
        self.max_message_size = max_message_size

        self.status = Status.not_started
        self.connected_clients = set()

        self._server_tasks_queues: dict[str, asyncio.Queue[communicator_pb2.MainMessage]] = defaultdict(
            lambda: asyncio.Queue()
        )
        self._main_tasks_queue: asyncio.Queue[communicator_pb2.MainMessage] = asyncio.Queue()
        self._tasks_futures = dict()


    @property
    def main_tasks_queue(self) -> asyncio.Queue:
        """ Return queue with tasks sent by VFL members. """
        return self._main_tasks_queue

    @property
    def tasks_queue(self) -> dict[str, asyncio.Queue]:
        """ Return dictionary of members` queues with associated scheduled tasks. """
        return self._server_tasks_queues

    @property
    def tasks_futures(self) -> dict:
        """ Return dictionary of tasks futures. """
        return self._tasks_futures

    async def start_servicer_and_server(self):
        """ Launch gRPC server and servicer. """
        server = grpc.aio.server(
            futures.ThreadPoolExecutor(max_workers=self.threadpool_max_workers),
            options=[
                ('grpc.max_send_message_length', self.max_message_size),
                ('grpc.max_receive_message_length', self.max_message_size)
            ]
        )
        communicator_pb2_grpc.add_CommunicatorServicer_to_server(self, server)
        server.add_insecure_port(f"{self.host}:{self.port}")  # TODO SSL goes here
        logger.info(f'Starting server at {self.host}:{self.port}')
        await server.start()
        await server.wait_for_termination()

    def process_heartbeat(self, request: communicator_pb2.HB) -> communicator_pb2.HB:
        """
        Process heartbeats from members. Indicate whether all members ready should start.
        """
        client_name = request.agent_name
        logger.debug(f"Got ping from client {client_name}")
        self.connected_clients.add(client_name)
        if len(self.connected_clients) == self.world_size:
            logger.info(f"All {self.world_size} clients connected")
            self.status = Status.all_ready
        else:
            self.status = Status.waiting
        return communicator_pb2.HB(
            agent_name=self.master_id,
            status=self.status,
        )

    async def Heartbeat(
            self,
            request_iterator: AsyncIterator[communicator_pb2.HB],
            context: grpc.aio.ServicerContext,
    ) -> AsyncIterator[communicator_pb2.HB]:
        """ Process heartbeats from member. Send world status back to member. """
        async for request in request_iterator:
            yield self.process_heartbeat(request)

    async def UnaryExchange(
            self, request: communicator_pb2.MainMessage, context: grpc.aio.ServicerContext
    ) -> communicator_pb2.MainMessage:
        """ Receive task response from member, return message indicating acknowledgment of the task received. """
        await self.main_tasks_queue.put(request)
        return communicator_pb2.MainMessage(
            message_type=MessageTypes.acknowledgment,
            task_id=request.task_id,
        )

    async def BidiExchange(
            self,
            request_iterator: AsyncIterator[communicator_pb2.MainMessage],
            context: grpc.aio.ServicerContext,
    ) -> None:
        """ Read members queries for task from master. Send task if scheduled in `self.tasks_queue`. """
        read = asyncio.create_task(self.process_requests(request_iterator, context))
        await read

    async def process_requests(
            self,
            request_iterator: AsyncIterator[communicator_pb2.MainMessage],
            context: grpc.aio.ServicerContext,
    ) -> None:
        """ Process requests for tasks, check the readiness of the task for the member, send the task if ready. """
        async for request in request_iterator:
            client_id = request.from_uid
            tasks_queue = self._server_tasks_queues.get(client_id)
            if tasks_queue is not None:
                try:
                    task_message = tasks_queue.get_nowait()
                    await context.write(task_message)
                    logger.debug(f'Sent task {task_message.method_name} to {client_id} ({task_message.task_id})')
                except asyncio.QueueEmpty:
                    pass
