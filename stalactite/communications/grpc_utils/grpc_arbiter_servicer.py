import asyncio
import logging
import time
from collections import defaultdict
from concurrent import futures

import grpc

from stalactite.communications.grpc_utils.generated_code import arbitered_communicator_pb2, \
    arbitered_communicator_pb2_grpc
from stalactite.communications.grpc_utils.utils import Status
from stalactite.ml.arbitered.base import Role

logger = logging.getLogger(__name__)
logging.getLogger('asyncio').setLevel(logging.ERROR)
logging.getLogger('grpc').setLevel(logging.ERROR)


class GRpcArbiterCommunicatorServicer(arbitered_communicator_pb2_grpc.ArbiteredCommunicatorServicer):
    def __init__(
            self,
            arbiter_id: str,
            port: str,
            world_size: int,
            threadpool_max_workers: int = 10,
            max_message_size: int = -1
    ):
        self.arbiter_id = arbiter_id
        self.port = port
        self.world_size = world_size
        self.threadpool_max_workers = threadpool_max_workers
        self.max_message_size = max_message_size

        self.host = '0.0.0.0'

        self._received_tasks = defaultdict(dict)
        self._tasks_to_send_queues = defaultdict(dict)
        self.connected_agents = dict()
        self.status = Status.not_started

        self.members: list = list()
        self.master = None

    async def get_from_tasks_to_send_dict(
            self, method_name: str, send_to_id: str, timeout: float = 30.0
    ) -> arbitered_communicator_pb2.MainArbiterMessage:
        timer_start = time.time()
        while (message := self._tasks_to_send_queues.get(send_to_id, dict()).pop(method_name, None)) is None:
            await asyncio.sleep(0.0)
            if time.time() - timer_start > timeout:
                raise TimeoutError(f"Could not send task: {method_name} to {send_to_id}.")
        return message

    async def start_servicer_and_server(self):
        """Launch gRPC server and servicer."""
        server = grpc.aio.server(
            futures.ThreadPoolExecutor(max_workers=self.threadpool_max_workers),
            options=[
                ("grpc.max_send_message_length", self.max_message_size),
                ("grpc.max_receive_message_length", self.max_message_size),
            ],
        )
        arbitered_communicator_pb2_grpc.add_ArbiteredCommunicatorServicer_to_server(self, server)
        server.add_insecure_port(f"{self.host}:{self.port}")
        logger.info(f"Starting server at {self.host}:{self.port}")
        self.status = Status.waiting
        await server.start()
        await server.wait_for_termination()

    def put_to_received_tasks(self, message: arbitered_communicator_pb2.MainArbiterMessage, receive_from_id: str):
        self._received_tasks[message.method_name][receive_from_id] = message

    async def SendToArbiter(
            self, request: arbitered_communicator_pb2.MainArbiterMessage, context: grpc.aio.ServicerContext
    ) -> arbitered_communicator_pb2.MainArbiterMessage:
        self.put_to_received_tasks(message=request, receive_from_id=request.sender_id)
        return arbitered_communicator_pb2.MainArbiterMessage(
            sender_id=self.arbiter_id,
        )

    async def RecvFromArbiter(
            self, request: arbitered_communicator_pb2.MainArbiterMessage, context: grpc.aio.ServicerContext
    ) -> arbitered_communicator_pb2.MainArbiterMessage:
        get_task = asyncio.create_task(
            self.get_from_tasks_to_send_dict(
                method_name=request.method_name, send_to_id=request.sender_id, timeout=request.get_response_timeout
            )
        )
        result = await get_task
        return result

    async def CheckIfAvailable(
            self, request: arbitered_communicator_pb2.IsReady, context: grpc.aio.ServicerContext
    ) -> arbitered_communicator_pb2.IsReady:
        self.connected_agents[request.sender_id] = time.time()
        self.status = Status.all_ready if len(self.connected_agents) == self.world_size + 1 else Status.waiting
        if request.role == Role.master:
            self.master = request.sender_id
        elif request.role == Role.member and request.sender_id not in self.members:
            self.members.append(request.sender_id)
        return arbitered_communicator_pb2.IsReady(
            sender_id=self.arbiter_id,
            ready=len(self.connected_agents) == self.world_size + 1
        )
