import asyncio
import argparse
from collections import defaultdict
from concurrent import futures
import logging
import time
from typing import AsyncIterator

import grpc
import torch

from gen_code import services_pb2, services_pb2_grpc
from helpers import PingResponse, format_important_logging
from utils import batch_generator, save_data, load_data, aggregate_tensors_list
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

        # TODO: state exchanges between asyncio.Task, do we need different instances? Other object / queue? (*)
        self._unary_unary_condition = asyncio.Condition()
        self._unary_unary_returned_clients = 0
        self._agg_results = defaultdict(dict)

        self._multi_multi_condition = asyncio.Condition()
        self._multi_multi_returned_clients = 0
        self._agg_results_batched = defaultdict(lambda: defaultdict(lambda: torch.tensor([])))

        self.connections = dict()
        asyncio.create_task(self._monitor_pings())

    async def _monitor_pings(self):
        """
        Monitor active connections.
        Remove clients which have not sent Ping message in `disconnect_idle_client_time` sec.
        """
        while True:
            await asyncio.sleep(3.)
            items = list(self.connections.items())
            for conn_id, last_ping in items:
                if time.time() - last_ping > self.disconnect_idle_client_time:
                    self.connections.pop(conn_id)
                    logger.info(f"Client {conn_id} disconnected")

    async def process_ping(self, request: services_pb2.Ping) -> services_pb2.Ping:
        """
        Process ping request from client. Send status indicating whether client should start.
        """
        client_name = request.data
        logger.debug(f"Got ping from client {client_name}")
        async with self._lock:
            self.connections[client_name] = time.time()

        # TODO: client start task logic in client?
        if len(self.connections) == self.world_size:
            logger.info(f"All {self.world_size} clients connected")
            msg_data = PingResponse.all_ready
        else:
            msg_data = PingResponse.waiting_for_other_connections
        return services_pb2.Ping(data=msg_data)

    async def aggregate_clients_results(
            self, request: services_pb2.SafetensorDataProto
    ) -> services_pb2.SafetensorDataProto:
        """
        Process client data, return aggregated clients response.
        """
        data = load_data(request.data) # TODO add numpy / tensor loading
        client_id = request.client_id
        client_iteration = request.iteration
        logger.info(f"Got aggregation query from {client_id}")
        async with self._unary_unary_condition:        # TODO (*)
            self._agg_results[client_iteration][client_id] = data
            self._unary_unary_condition.notify_all()
        return await self._aggregate_unary_results(client_iteration, client_id)

    async def _aggregate_unary_results(self, iteration: int, client_id: str) -> services_pb2.SafetensorDataProto:
        """
        Aggregate clients results. Send aggregated data back to clients.
        """
        async with self._unary_unary_condition:        # TODO (*)
            await self._unary_unary_condition.wait_for(
                lambda: len(self._agg_results[iteration]) == self.world_size
            )
        start = time.time()
        collected_tensors = self._agg_results[iteration].values()
        returned_tensor = aggregate_tensors_list(collected_tensors)
        logger.info(format_important_logging(
            f"All queries collected, returning aggregated result. Aggregation time: {round(time.time() - start, 4)}")
        )

        async with self._lock:
            self._unary_unary_returned_clients += 1
            if self._unary_unary_returned_clients == self.world_size:
                self._agg_results[iteration] = dict()
                self._unary_unary_returned_clients = 0

        return services_pb2.SafetensorDataProto(
            data=save_data(returned_tensor),
            client_id=client_id,
            iteration=iteration,
        )

    async def aggregate_clients_batched_results(
            self, request_iterator: AsyncIterator[services_pb2.SafetensorDataProto], context: grpc.aio.ServicerContext
    ) -> None:
        """
        Read clients batched streams. Stream aggregated data in batches back to clients.
        """
        read = asyncio.create_task(self._collect_batched_requests(request_iterator))
        client_id, client_iteration = await read
        write = asyncio.create_task(self._send_batched_response(context, client_id, client_iteration))
        await write

    async def _collect_batched_requests(self, request_iterator: AsyncIterator[services_pb2.SafetensorDataProto]):
        """
        Process client stream. Add data batches to aggregator.
        """
        client_id, client_iteration = None, None
        async for request in request_iterator:
            request_data = load_data(request.data)
            client_id = request.client_id
            client_iteration = request.iteration
            batch_num = request.batch
            total_batches = request.total_batches
            async with self._lock:
                self._agg_results_batched[client_iteration][client_id] = torch.cat(
                    [self._agg_results_batched[client_iteration][client_id], request_data]
                )
            if batch_num == total_batches - 1:
                async with self._multi_multi_condition:
                    self._multi_multi_condition.notify_all()
        return client_id, client_iteration

    async def _send_batched_response(self, context: grpc.aio.ServicerContext, client_id: str, iteration: int):
        """
        Create response stream after all clients done writing. Stream data back to clients.
        """
        async with self._multi_multi_condition:
            await self._multi_multi_condition.wait_for(
                lambda: len(self._agg_results_batched[iteration]) == self.world_size
            )
        collected_tensors = self._agg_results_batched[iteration].values()
        returned_tensor = aggregate_tensors_list(collected_tensors)

        for batched_data in batch_generator(returned_tensor, self.batch_size):
            await context.write(services_pb2.SafetensorDataProto(
                data=save_data(batched_data.data),
                client_id=client_id,
                iteration=iteration,
                batch=batched_data.batch,
                total_batches=batched_data.total_batches,
            ))
        async with self._lock:
            self._multi_multi_returned_clients += 1
            if self._multi_multi_returned_clients == self.world_size:
                self._agg_results_batched[iteration] = defaultdict(lambda: torch.tensor([]))
                self._multi_multi_returned_clients = 0


class GRpcCommunicatorServicer(services_pb2_grpc.CommunicatorServicer):
    def __init__(
            self, world_size: int, *args, disconnect_idle_client_time: float = 6., batch_size: int = 1_000, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.comm_manager = CommunicationManager(
            world_size=world_size,
            disconnect_idle_client_time=disconnect_idle_client_time,
            batch_size=batch_size
        )

    async def PingPong(
            self, request_iterator: AsyncIterator[services_pb2.Ping], context: grpc.aio.ServicerContext
    ) -> AsyncIterator[services_pb2.Ping]:
        """
        Perform bidirectional ping-pong streaming.
        Wait for all the clients to be connected. Send status ClientStatus in services_pb2.Ping data.
        """
        async for request in request_iterator:
            returned_status = await self.comm_manager.process_ping(request)
            yield returned_status

    async def ExchangeBinarizedDataUnaryUnary(
            self, request: services_pb2.SafetensorDataProto, context: grpc.aio.ServicerContext
    ) -> services_pb2.SafetensorDataProto:
        """
        Perform unary-unary torch.Tensor exchange with aggregation between clients.
        Collect data from all clients. Send aggregated result.
        """
        return await self.comm_manager.aggregate_clients_results(request)

    async def ExchangeBinarizedDataStreamStream(
            self, request_iterator: AsyncIterator[services_pb2.SafetensorDataProto], context: grpc.aio.ServicerContext
    ) -> None:
        """
        Perform multi-multi torch.Tensor batched exchange with aggregation between clients.
        Collect batched data from all clients. Send aggregated result in batches.
        """
        await self.comm_manager.aggregate_clients_batched_results(request_iterator, context)


async def serve(
        port: str,
        max_workers: int,
        max_message_size: int,
        world_size: int,
        batch_size: int,
):
    server = grpc.aio.server(
        futures.ThreadPoolExecutor(max_workers=max_workers),
        options=[
            ('grpc.max_send_message_length', max_message_size),
            ('grpc.max_receive_message_length', max_message_size)
        ]
    )
    services_pb2_grpc.add_CommunicatorServicer_to_server(
        GRpcCommunicatorServicer(world_size=world_size, batch_size=batch_size),
        server
    )
    server.add_insecure_port("[::]:" + port)
    await server.start()
    logger.info(f"Server started, listening on {port}")
    await server.wait_for_termination()


if __name__ == "__main__":
    parser.add_argument("-w", "--world_size", type=int, help="Number of clients")
    parser.add_argument("--port", type=str, default="50051", help="Server port")
    parser.add_argument("--batch_size", type=int, default=1_000, help="Server port")
    parser.add_argument("--max_message_size", type=int, default=MAX_MESSAGE_LENGTH, help="Message size (bytes)")
    parser.add_argument("--thread_pool_max_workers", type=int, default=20, help="Max ThreadPoolExecutor workers")
    args = parser.parse_args()
    try:
        asyncio.get_event_loop().run_until_complete(serve(
            port=args.port,
            max_workers=args.thread_pool_max_workers,
            max_message_size=args.max_message_size,
            world_size=args.world_size,
            batch_size=args.batch_size,
        ))
    except KeyboardInterrupt:
        logger.info('Terminated')
