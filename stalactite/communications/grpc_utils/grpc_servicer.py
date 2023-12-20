import asyncio
import concurrent
from collections import defaultdict
import enum
import time
import uuid
from abc import ABC, abstractmethod
# from queue import Queue, Empty
from dataclasses import dataclass
import logging
from typing import Optional, Dict, Union, AsyncIterator, List, Any
import threading
from concurrent import futures

import grpc
import torch
import safetensors.torch

from stalactite.base import (
    PartyMaster,
    PartyMember,
    PartyCommunicator,
    ParticipantFuture,
    Party,
    PartyDataTensor,
    RecordsBatch,
)

from stalactite.communications.grpc_utils.generated_code import communicator_pb2, communicator_pb2_grpc
from stalactite.communications.grpc_utils.utils import Status, EventTask

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
sh = logging.StreamHandler()
sh.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(sh)



class GRpcCommunicatorServicer(communicator_pb2_grpc.CommunicatorServicer):
    connected_clients = set()

    def __init__(
            self,
            world_size: int,
            master_id: str,
            host: str,
            port: str,
            *args,
            threadpool_max_workers: int = 10,
            max_message_size: int = -1,
            **kwargs):
        super().__init__(*args, **kwargs)
        self.world_size = world_size
        self.master_id = master_id
        self.host = host
        self.port = port
        self.threadpool_max_workers = threadpool_max_workers
        self.max_message_size = max_message_size

        self.status = Status.not_started

        self._lock = asyncio.Lock()

        self._server_tasks_queues: dict[str, asyncio.Queue[EventTask]] = defaultdict(lambda: asyncio.Queue())


        #???????????
        self._main_tasks_queue: asyncio.Queue[EventTask] = asyncio.Queue()



        self._tasks_futures = dict()

        self._client_contexts = dict()

    @property
    def main_tasks_queue(self) -> asyncio.Queue:
        return self._main_tasks_queue

    @property
    def tasks_queue(self) -> dict[str, asyncio.Queue]:
        return self._server_tasks_queues

    @property
    def tasks_futures(self) -> dict:
        return self._tasks_futures

    async def start_servicer_and_server(self):
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
        Process heartbeats from members. Send status indicating whether all members ready should start.
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
        async for request in request_iterator:
            yield self.process_heartbeat(request)

    async def UnaryExchange(
            self, request: communicator_pb2.MainMessage, context: grpc.aio.ServicerContext
    ) -> communicator_pb2.MainMessage:
        await self.main_tasks_queue.put(request)
        return communicator_pb2.MainMessage(
            message_type='ack',
            task_id=request.task_id,
        )

    async def BidiExchange(
            self,
            request_iterator: AsyncIterator[communicator_pb2.MainMessage],
            context: grpc.aio.ServicerContext,
    ) -> None:
        read = asyncio.create_task(self._process_requests(request_iterator, context))
        write = asyncio.create_task(self._generate_responses(context))

        await read
        await write

    async def _process_requests(
            self,
            request_iterator: AsyncIterator[communicator_pb2.MainMessage],
            context: grpc.aio.ServicerContext,
    ) -> None:
        async for request in request_iterator:
            context.set_details(request.from_uid)

    async def _generate_responses(self, context: grpc.aio.ServicerContext):
        await asyncio.sleep(0.1)
        while True:
            client_id = context.details()
            tasks_queue = self._server_tasks_queues.get(client_id)
            if tasks_queue is not None:
                try:
                    task = tasks_queue.get_nowait()
                    if task.send_to_id != client_id:
                        raise RuntimeError(f'Tried to sent task with receiver id: {task.send_to_id} to {client_id}')
                    await context.write(task.message)
                    logger.debug(f'Sent task {task.message} to {task.send_to_id} ({task.task_id})')
                except asyncio.QueueEmpty:
                    await asyncio.sleep(1.)
            else:
                await asyncio.sleep(1.)
