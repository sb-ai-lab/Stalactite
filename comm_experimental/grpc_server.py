from collections import defaultdict
from concurrent import futures
import logging
import time
from typing import Iterator, ValuesView
from threading import Thread

import grpc
import numpy as np
import torch
import safetensors.torch

from gen_code import services_pb2, services_pb2_grpc
from helpers import PingResponse, batch_generator

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
sh = logging.StreamHandler()
sh.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(sh)


MAX_MESSAGE_LENGTH = 8 * 1_000_000 * 10 + 5_000_000

class CommunicationManager:
    def __init__(self, world_size: int, disconnect_idle_client_time: float = 6., batch_size: int = 100):

        self.world_size = world_size
        self.disconnect_idle_client_time = disconnect_idle_client_time
        self.batch_size = batch_size

        self.connections = dict()
        self._agg_results = defaultdict(dict)
        self._agg_results_batched = defaultdict(lambda: defaultdict(lambda: torch.tensor([])))

        self.connections_checker = Thread(target=self.monitor_pings)
        self.connections_checker.daemon = True
        self.connections_checker.start()

    def monitor_pings(self):
        while True:
            items = list(self.connections.items())
            for conn_id, last_ping in items:
                if time.time() - last_ping > self.disconnect_idle_client_time:
                    self.connections.pop(conn_id)
                    logger.debug(f"Client disconnected: {conn_id}")
                else:
                    time.sleep(3.)

    # def start(self):
    #
    #     # self._close()

    # def stop(self):
    #     self.connections_checker.join(timeout=0.)
    #     logger.info("Stopped ping checking thread.")

    def process_ping(self, request: services_pb2.Ping) -> services_pb2.Ping:
        client_name = request.data
        logger.debug(f"Got ping from client {client_name}")
        self.connections[client_name] = time.time() # TODO add monitoring of last update time
        if len(self.connections) == self.world_size:
            logger.info(f"All {self.world_size} clients connected")
            msg_data = PingResponse.all_ready
        else:
            msg_data = PingResponse.waiting_for_other_connections
        return services_pb2.Ping(data=msg_data)


    def _aggregare_tensors_list(self, values: ValuesView[torch.Tensor]) -> torch.Tensor:
        return torch.sum(torch.stack(list(values)), dim=0)


    def aggregate_clients_results(self, request: services_pb2.SafetensorDataProto) -> services_pb2.SafetensorDataProto:
        # TODO agg logic fix
        request_data = safetensors.torch.load(request.data)
        client_id = request.client_id
        client_iteration = request.iteration
        logger.info(f"Got aggregation query from {client_id}")
        self._agg_results[client_iteration][client_id] = request_data['tensor']
        while len(self._agg_results[client_iteration]) != self.world_size:
            continue

        returned_tensor = self._aggregare_tensors_list(self._agg_results[client_iteration].values())
        self._agg_results[client_iteration] = dict()
        logger.info(f"All queries collected, returning aggregated result")
        return services_pb2.SafetensorDataProto(
            data=safetensors.torch.save(tensors={'tensor': returned_tensor}),
            client_id=client_id,
            iteration=client_iteration,
        )

    def aggregate_batched_clients_results(
            self, request_iterator: Iterator[services_pb2.SafetensorDataProto]
    ) -> Iterator[services_pb2.SafetensorDataProto]:
        for request in request_iterator:
            print('request.client_id', request.client_id)
            request_data = safetensors.torch.load(request.data)
            client_id = request.client_id
            client_iteration = request.iteration
            batch_num = request.batch
            total_batches = request.total_batches
            self._agg_results_batched[client_iteration][client_id] = torch.cat(
                [self._agg_results_batched[client_iteration][client_id], request_data['tensor']]
            )
            if batch_num == total_batches - 1:
                while len(self._agg_results_batched[client_iteration]) != self.world_size:
                    continue
                returned_tensor = self._aggregare_tensors_list(self._agg_results_batched[client_iteration].values())

                logger.info(f"All queries collected, returning aggregated result")
                for batched_data in batch_generator(returned_tensor, self.batch_size):
                    yield services_pb2.SafetensorDataProto(
                        data=safetensors.torch.save(tensors={'tensor': batched_data.data}),
                        client_id=client_id,
                        iteration=client_iteration,
                        batch=batched_data.batch,
                        total_batches=batched_data.total_batches,
                    )
                self._agg_results_batched[client_iteration] = defaultdict(lambda: torch.tensor([]))

class GRpcCommunicatorServicer(services_pb2_grpc.CommunicatorServicer):
    def __init__(self, comm_manager: CommunicationManager, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.comm_manager = comm_manager
        # self.comm_manager.start()

    def PingPong(
            self, request_iterator: Iterator[services_pb2.Ping], context: grpc.ServicerContext
    ) -> Iterator[services_pb2.Ping]:
        for request in request_iterator:
            returned_status = self.comm_manager.process_ping(request)
            yield returned_status

    def ExchangeBinarizedDataUnaryUnary(
            self, request: services_pb2.SafetensorDataProto, context: grpc.ServicerContext
    ) -> services_pb2.SafetensorDataProto:
        result = self.comm_manager.aggregate_clients_results(request)
        return result
        #
        # return services_pb2.SafetensorDataProto(
        #     data=safetensors.torch.save({'tensor': torch.tensor([1, 2], dtype=torch.float32)})
        # )

    def ExchangeBinarizedDataStreamStream(
            self, request_iterator: Iterator[services_pb2.SafetensorDataProto], context: grpc.ServicerContext
    ) -> Iterator[services_pb2.SafetensorDataProto]:
        print('ExchangeBinarizedDataStreamStream', context.peer())
        result_batches = self.comm_manager.aggregate_batched_clients_results(request_iterator)
        for result in result_batches:
            print(context.peer())
            yield result

def serve():
    try:
        port = "50051"
        server = grpc.server(
            futures.ThreadPoolExecutor(max_workers=20),
            options=[
                ('grpc.max_send_message_length', MAX_MESSAGE_LENGTH),
                ('grpc.max_receive_message_length', MAX_MESSAGE_LENGTH)
            ]
        )
        comm_manager = CommunicationManager(world_size=2)
        services_pb2_grpc.add_CommunicatorServicer_to_server(GRpcCommunicatorServicer(comm_manager=comm_manager), server)
        server.add_insecure_port("[::]:" + port)
        server.start()
        # comm_manager.start()
        print("Server started, listening on " + port)
        server.wait_for_termination()
    except KeyboardInterrupt:
        # comm_manager.stop()
        print('Terminated')


if __name__ == "__main__":
    serve()
