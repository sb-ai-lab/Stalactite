import asyncio
import argparse
from collections import defaultdict
from concurrent import futures
import logging
import time
from typing import Iterator, ValuesView, AsyncIterator
from threading import Thread

import grpc
import numpy as np
import torch
import safetensors.torch

from gen_code import services_pb2, services_pb2_grpc
from helpers import PingResponse, batch_generator, format_important_logging
from constants import MAX_MESSAGE_LENGTH

parser = argparse.ArgumentParser()

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
sh = logging.StreamHandler()
sh.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(sh)


class CommunicationManager:
    def __init__(self, world_size: int, disconnect_idle_client_time: float = 6., batch_size: int = 100):

        self.world_size = world_size
        self.disconnect_idle_client_time = disconnect_idle_client_time
        self.batch_size = batch_size

        self._lock = asyncio.Lock()
        self._unary_unary_condition = asyncio.Condition()

        self._returned_clients = 0

        self.connections = dict()
        self._agg_results = defaultdict(dict)
        self._agg_results_batched = defaultdict(lambda: defaultdict(lambda: torch.tensor([])))

        asyncio.create_task(self.monitor_pings())

    async def monitor_pings(self):
        while True:
            await asyncio.sleep(3.)
            items = list(self.connections.items())
            for conn_id, last_ping in items:
                if time.time() - last_ping > self.disconnect_idle_client_time:
                    self.connections.pop(conn_id)
                    logger.info(f"Client {conn_id} disconnected")

    async def process_ping(self, request: services_pb2.Ping) -> services_pb2.Ping:
        client_name = request.data
        logger.debug(f"Got ping from client {client_name}")
        # async with self._lock:
        self.connections[client_name] = time.time()  # TODO add monitoring of last update time
        if len(self.connections) == self.world_size:
            logger.info(f"All {self.world_size} clients connected")
            msg_data = PingResponse.all_ready
        else:
            msg_data = PingResponse.waiting_for_other_connections
        return services_pb2.Ping(data=msg_data)

    @staticmethod
    def _aggregate_tensors_list(values: ValuesView[torch.Tensor]) -> torch.Tensor:
        return torch.sum(torch.stack(list(values)), dim=0)

    @staticmethod
    def _load_data(data: bytes) -> torch.Tensor:
        return safetensors.torch.load(data)['tensor']

    @staticmethod
    def _save_data(data: torch.Tensor) -> bytes:
        return safetensors.torch.save(tensors={'tensor': data})

    async def aggregate_clients_results(
            self, request: services_pb2.SafetensorDataProto
    ) -> services_pb2.SafetensorDataProto:
        data = self._load_data(request.data)
        client_id = request.client_id
        client_iteration = request.iteration
        logger.info(f"Got aggregation query from {client_id}")
        async with self._unary_unary_condition:
            self._agg_results[client_iteration][client_id] = data
            self._unary_unary_condition.notify_all()
        return await self._aggregate_unary_results(client_iteration, client_id)

    async def _aggregate_unary_results(self, iteration: int, client_id: str) -> services_pb2.SafetensorDataProto:
        async with self._unary_unary_condition:
            await self._unary_unary_condition.wait_for(
                lambda: len(self._agg_results[iteration]) == self.world_size
            )
        start = time.time()
        collected_tensors = self._agg_results[iteration].values()
        returned_tensor = self._aggregate_tensors_list(collected_tensors)
        logger.info(format_important_logging(
            f"All queries collected, returning aggregated result. Aggregation time: {round(time.time() - start, 4)}")
        )

        async with self._lock:
            self._returned_clients += 1
            if self._returned_clients == self.world_size:
                self._agg_results[iteration] = dict()

        return services_pb2.SafetensorDataProto(
            data=self._save_data(returned_tensor),
            client_id=client_id,
            iteration=iteration,
        )

    #
    # def aggregate_batched_clients_results(
    #         self, request_iterator: Iterator[services_pb2.SafetensorDataProto]
    # ) -> Iterator[services_pb2.SafetensorDataProto]:
    #     for request in request_iterator:
    #         print('request.client_id', request.client_id)
    #         request_data = safetensors.torch.load(request.data)
    #         client_id = request.client_id
    #         client_iteration = request.iteration
    #         batch_num = request.batch
    #         total_batches = request.total_batches
    #         self._agg_results_batched[client_iteration][client_id] = torch.cat(
    #             [self._agg_results_batched[client_iteration][client_id], request_data['tensor']]
    #         )
    #         if (len(self._agg_results_batched[client_iteration]) == self.world_size) and (batch_num == total_batches - 1):
    #             returned_tensor = self._aggregare_tensors_list(self._agg_results_batched[client_iteration].values())
    #             # self._agg_results_batched[client_iteration] = defaultdict(lambda: torch.tensor([]))
    #             for batched_data in batch_generator(returned_tensor, self.batch_size):
    #                 yield services_pb2.SafetensorDataProto(
    #                     data=safetensors.torch.save(tensors={'tensor': batched_data.data}),
    #                     client_id=client_id,
    #                     iteration=client_iteration,
    #                     batch=batched_data.batch,
    #                     total_batches=batched_data.total_batches,
    #                 )
    #
    #


class GRpcCommunicatorServicer(services_pb2_grpc.CommunicatorServicer):
    def __init__(
            self, world_size: int, *args, disconnect_idle_client_time: float = 6., batch_size: int = 100, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.comm_manager = CommunicationManager(
            world_size=world_size,
            disconnect_idle_client_time=disconnect_idle_client_time,
            batch_size=batch_size
        )

    async def PingPong(
            self, request_iterator: AsyncIterator[services_pb2.Ping], context: grpc.ServicerContext
    ) -> AsyncIterator[services_pb2.Ping]:
        async for request in request_iterator:
            returned_status = await self.comm_manager.process_ping(request)
            yield returned_status

    async def ExchangeBinarizedDataUnaryUnary(
            self, request: services_pb2.SafetensorDataProto, context: grpc.ServicerContext
    ) -> services_pb2.SafetensorDataProto:
        return await self.comm_manager.aggregate_clients_results(request)

    # def ExchangeBinarizedDataStreamStream(
    #         self, request_iterator: Iterator[services_pb2.SafetensorDataProto], context: grpc.ServicerContext
    # ) -> Iterator[services_pb2.SafetensorDataProto]:
    #     print('ExchangeBinarizedDataStreamStream', context.peer())
    #     result_batches = self.comm_manager.aggregate_batched_clients_results(request_iterator)
    #     # returned_tensor, batch_size, client_id, client_iteration = self.comm_manager.aggregate_batched_clients_results(request_iterator)
    #
    #     logger.info(f"All queries collected, returning aggregated result")
    #     # for batched_data in batch_generator(returned_tensor, batch_size):
    #     #     yield services_pb2.SafetensorDataProto(
    #     #         data=safetensors.torch.save(tensors={'tensor': batched_data.data}),
    #     #         client_id=client_id,
    #     #         iteration=client_iteration,
    #     #         batch=batched_data.batch,
    #     #         total_batches=batched_data.total_batches,
    #     #     )
    #     for result in result_batches:
    #         print(context.peer())
    #         yield result


async def serve(
        port: str,
        max_workers: int,
        max_message_size: int,
        world_size: int,
):
    server = grpc.aio.server(
        futures.ThreadPoolExecutor(max_workers=max_workers),
        options=[
            ('grpc.max_send_message_length', max_message_size),
            ('grpc.max_receive_message_length', max_message_size)
        ]
    )
    services_pb2_grpc.add_CommunicatorServicer_to_server(
        GRpcCommunicatorServicer(world_size=world_size),
        server
    )
    server.add_insecure_port("[::]:" + port)
    await server.start()
    logger.info(f"Server started, listening on {port}")
    await server.wait_for_termination()


if __name__ == "__main__":
    parser.add_argument("-w", "--world_size", type=int, help="Number of clients")
    parser.add_argument("--port", type=str, default="50051", help="Server port")
    parser.add_argument("--max_message_size", type=int, default=MAX_MESSAGE_LENGTH, help="Message size (bytes)")
    parser.add_argument("--thread_pool_max_workers", type=int, default=20, help="Max ThreadPoolExecutor workers")
    args = parser.parse_args()
    try:
        asyncio.get_event_loop().run_until_complete(serve(
            port=args.port,
            max_workers=args.thread_pool_max_workers,
            max_message_size=args.max_message_size,
            world_size=args.world_size,
        ))
    except KeyboardInterrupt:
        logger.info('Terminated')
