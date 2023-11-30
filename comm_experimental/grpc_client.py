import argparse
import asyncio
import logging
import time
from typing import Any, AsyncIterator
import uuid

import grpc
import numpy as np
import torch
import safetensors.torch

from gen_code import services_pb2, services_pb2_grpc
from utils import batch_generator, save_data, load_data, BatchedData
from helpers import PingResponse, ClientStatus, ClientTask, format_important_logging, Serialization
from constants import MAX_MESSAGE_LENGTH

parser = argparse.ArgumentParser()

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
sh = logging.StreamHandler()
sh.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(sh)


def batch_generator():
    total_rows = 1_000
    num_columns = 10
    batch_size = 10
    total_batches = int(np.ceil(total_rows / batch_size))
    for batch in range(total_batches):
        yield BatchedData(
            data=torch.rand(total_rows // batch_size, num_columns, dtype=torch.float64),
            batch=batch,
            total_batches=total_batches,
        )


class GRpcClient:
    def __init__(
            self,
            server_host: str,
            server_port: int | str,
            ping_interval: float = 5.,
            unary_unary_operations_timeout: float = 5000.,
            max_message_size: int = MAX_MESSAGE_LENGTH,
            num_rows: int = 10_000,
            num_cols: int = 10,
    ):
        self.ping_interval = ping_interval
        self.num_rows = num_rows
        self.num_cols = num_cols

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

    async def start(self):
        try:
            logger.info("Starting ping-pong with the server")
            pingpong_responses = self._stub.PingPong(
                self.ping_messages_generator(),
                wait_for_ready=True,
            )
            self._pings_task = asyncio.create_task(self.process_pongs(server_response_iterator=pingpong_responses))
            # await self.run_task(ClientTask.batched_exchange)
            await self.run_task(ClientTask.exchange_tensor)
        except KeyboardInterrupt:
            pass

    @property
    def iteration(self):
        return self._iteration

    @property
    def client_name(self):
        return str(self._uuid)

    @property
    def ping_message(self):
        return services_pb2.Ping(data=self.client_name)

    def tensor_message(self, data: BatchedData) -> services_pb2.SafetensorDataProto:
        return services_pb2.SafetensorDataProto(
            data=save_data(data.data),
            iteration=self.iteration,
            client_id=self.client_name,
            batch=data.batch,
            total_batches=data.total_batches
        )

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

    def _get_future(self, ):
        start = time.time()
        future = self._stub.ExchangeBinarizedDataUnaryUnary(
            services_pb2.SafetensorDataProto(
                data=safetensors.torch.save(
                    tensors={'tensor': torch.rand(self.num_rows, self.num_cols, dtype=torch.float64)},
                ),
                client_id=self.client_name,
                iteration=self.iteration,
            ),
            timeout=self.unary_unary_operations_timeout,
        )
        logger.info(format_important_logging(
            f"Message was created (tensor generation, serialization), in {round(time.time() - start, 4)}"
        ))
        return future

    def _exchange_messages_generator(self, ):
        for batch in batch_generator():
            yield self.tensor_message(batch)

    def _get_iter_future(self):
        return self._stub.ExchangeBinarizedDataStreamStream(
            self._exchange_messages_generator(),
            wait_for_ready=True,
        )

    def get_task(self, task_type: ClientTask):
        if task_type == ClientTask.exchange_tensor:
            return self._get_future()
        elif task_type == ClientTask.batched_exchange_tensor:
            return self._get_iter_future()
        else:
            raise ValueError(f'Task type {task_type} not known')

    async def get_task_result(self, task_type: ClientTask, future: Any) -> torch.Tensor:
        result = torch.tensor([])
        if task_type == ClientTask.exchange_tensor:
            start = time.time()
            data = await future
            breakpoint = time.time()
            result = load_data(data, serialization=Serialization.safetensors)
            end = time.time()
            total_time = end - start
            awaiting_time = breakpoint - start
            deserialization_time = end - breakpoint
            logger.info(format_important_logging(
                f"Result got in {round(total_time, 4)} sec: \n"
                f" - coro awaited for {round(awaiting_time, 4)} sec;\n"
                f" - deserialization time {round(deserialization_time, 4)}"
            ))
        elif task_type == ClientTask.batched_exchange_tensor:
            start = time.time()
            async for batch in future:
                data_batch = load_data(batch, serialization=Serialization.safetensors)
                result = torch.cat([result, data_batch])
            end = time.time()
            logger.info(format_important_logging(
                f"Result got in {round(end - start, 4)} sec: \n"
            ))
        else:
            raise ValueError(f'Task type {task_type} not known')
        return result

    async def run_task(self, task_type: ClientTask):
        async with self._condition:
            await self._condition.wait_for(lambda: self._status == ClientStatus.active)
        start = time.time()
        task = self.get_task(task_type)
        result = await self.get_task_result(task_type, task)
        full_time = time.time() - start
        logger.info(format_important_logging(
            f"Got aggregation result from server: result shape {result.shape}, result[0, 0] {result[0, 0]}\n"
            f"Full execution time: {round(full_time, 4)} sec"
        ))

    async def _close(self):
        self._pings_task.cancel()
        await self._grpc_channel.close()


if __name__ == '__main__':
    parser.add_argument("--port", type=str, default="50051", help="Server port")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Server host")
    parser.add_argument("--max_message_size", type=int, default=MAX_MESSAGE_LENGTH, help="Message size (bytes)")
    parser.add_argument("--thread_pool_max_workers", type=int, default=20, help="Max ThreadPoolExecutor workers")
    parser.add_argument("--ping_interval", type=float, default=1., help="Ping interval (sec)")
    parser.add_argument("--num_rows", type=int, default=10_000, help="Number of tensor rows")
    parser.add_argument("--num_cols", type=int, default=10, help="Number of tensor rows")
    args = parser.parse_args()

    client = GRpcClient(
        server_host=args.host,
        server_port=args.port,
        ping_interval=args.ping_interval,
        max_message_size=args.max_message_size,
        num_rows=args.num_rows,
        num_cols=args.num_cols,
    )
    asyncio.get_event_loop().run_until_complete(client.start())
