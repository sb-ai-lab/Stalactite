import argparse
import asyncio
import logging
import time
from typing import AsyncIterator
import uuid
from contextlib import asynccontextmanager

import grpc
import torch

from generated_code import services_pb2, services_pb2_grpc
from utils import (
    PingResponse,
    ClientStatus,
    ClientTask,
    format_important_logging,
    Serialization,
    MAX_MESSAGE_LENGTH,
    Task,
    ExperimentalData,
)


parser = argparse.ArgumentParser()

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
sh = logging.StreamHandler()
sh.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(sh)


class GRpcClient:
    def __init__(
            self,
            server_host: str,
            server_port: int | str,
            ping_interval: float = 5.,
            unary_unary_operations_timeout: float = 5000.,
            max_message_size: int = MAX_MESSAGE_LENGTH,
    ):
        self.ping_interval = ping_interval
        self._task_queue = asyncio.Queue()

        self.unary_unary_operations_timeout = unary_unary_operations_timeout

        self._uuid = uuid.uuid4()
        self._grpc_channel = grpc.aio.insecure_channel(
            f"{server_host}:{server_port}",
            options=[
                ('grpc.max_send_message_length', max_message_size),
                ('grpc.max_receive_message_length', max_message_size),
            ]
        )
        self._stub = services_pb2_grpc.CommunicatorStub(self._grpc_channel)

        self._status = ClientStatus.waiting
        self._condition = asyncio.Condition()
        self._iteration = 1
        self._pings_task = None

        logger.info(f"Initialized {self._uuid} client: ({server_host}:{server_port})")

    async def add_task_to_queue(self, task: Task):
        await self._task_queue.put(task)

    @asynccontextmanager
    async def start(self):
        try:
            logger.info("Starting ping-pong with the server")
            pingpong_responses = self._stub.PingPong(
                self.ping_messages_generator(),
                wait_for_ready=True,
            )
            self._pings_task = asyncio.create_task(self.process_pongs(server_response_iterator=pingpong_responses))
            self._run_task = asyncio.create_task(self.run_task())
            yield self
            await self._run_task
            # await self.run_task(ClientTask.exchange)
        # except KeyboardInterrupt:
        #     pass
        finally:
            await self._run_task
            await self._close()

    @property
    def iteration(self):
        return self._iteration

    @property
    def client_name(self):
        return str(self._uuid)

    @property
    def ping_message(self):
        return services_pb2.Ping(data=self.client_name)

    async def ping_messages_generator(self):
        while True:
            await asyncio.sleep(self.ping_interval)
            yield self.ping_message

    async def process_pongs(self, server_response_iterator: AsyncIterator[services_pb2.Ping]):
        try:
            async for response in server_response_iterator:
                logger.debug(f"Got pong from server: {response.data}")
                message = response.data
                if message == PingResponse.waiting_for_other_connections:
                    self._status = ClientStatus.waiting
                elif message == PingResponse.all_ready:
                    logger.info("Got `all ready` pong from server")
                    self._status = ClientStatus.active
                async with self._condition:
                    self._condition.notify_all()
        except Exception as exc:
            logger.error(f'Exception: {exc}')

    async def _process_task(self, task: Task):
        start = time.time()
        future = task.start_rpc(stub=self._stub, client_name=self.client_name, client_iteration=self.iteration)
        result = await task.collect_results_rpc(future)
        full_time = time.time() - start
        logger.info(format_important_logging(
            f"Got aggregation result from server: result shape {result.shape}, result[0, 0] {result[0, 0]}\n"
            f"Full execution time: {round(full_time, 4)} sec"
        ))

    async def run_task(self):
        async with self._condition:
            await self._condition.wait_for(lambda: self._status == ClientStatus.active)
        while True:
            task = await self._task_queue.get()
            if task.type == ClientTask.finish:
                logger.info('Got task `ClientTask.finish`. Terminating')
                return
            await self._process_task(task)

    async def _close(self):
        self._pings_task.cancel()
        await self._grpc_channel.close()


async def run_client(args):
    client = GRpcClient(
        server_host=args.host,
        server_port=args.port,
        ping_interval=args.ping_interval,
        max_message_size=args.max_message_size,
    )

    data_tensor = ExperimentalData(
        num_rows=args.num_rows,
        num_columns=args.num_cols,
        dtype=torch.float64,
        batch_size=args.batch_size,
        serialization=Serialization.safetensors,
    )

    data_proto = ExperimentalData(
        num_rows=args.num_rows,
        num_columns=args.num_cols,
        dtype=torch.float64,
        batch_size=args.batch_size,
        serialization=Serialization.protobuf,
    )

    async with client.start() as party:
        await party.add_task_to_queue(Task(task_type=ClientTask.exchange_array, data=data_proto))
        await party.add_task_to_queue(Task(task_type=ClientTask.batched_exchange_array, data=data_proto))
        await party.add_task_to_queue(Task(task_type=ClientTask.exchange_tensor, data=data_tensor))
        await party.add_task_to_queue(Task(task_type=ClientTask.batched_exchange_tensor, data=data_tensor))

        await party.add_task_to_queue(Task(task_type=ClientTask.finish))


if __name__ == '__main__':
    parser.add_argument("--port", type=str, default="50051", help="Server port")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Server host")
    parser.add_argument("--max_message_size", type=int, default=MAX_MESSAGE_LENGTH, help="Message size (bytes)")
    parser.add_argument("--thread_pool_max_workers", type=int, default=20, help="Max ThreadPoolExecutor workers")
    parser.add_argument("--ping_interval", type=float, default=1., help="Ping interval (sec)")
    parser.add_argument("--num_rows", type=int, default=100_000, help="Number of tensor rows")
    parser.add_argument("--num_cols", type=int, default=10, help="Number of tensor rows")
    parser.add_argument("--batch_size", type=int, default=1000, help="Number of tensor rows in batch")
    args = parser.parse_args()

    asyncio.get_event_loop().run_until_complete(run_client(args))
