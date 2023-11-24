from collections import defaultdict
from concurrent import futures
import logging
import time
from typing import Iterator
from threading import Thread

import grpc
import torch
import safetensors.torch

from gen_code import services_pb2, services_pb2_grpc
from helpers import PingResponse

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
sh = logging.StreamHandler()
sh.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(sh)


class CommunicationManager:
    def __init__(self, world_size: int, disconnect_idle_client_time: float = 6.):

        self.world_size = world_size
        self.disconnect_idle_client_time = disconnect_idle_client_time

        self.connections = dict()
        self._agg_results = defaultdict(dict)

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

    def aggregate_clients_results(self, request: services_pb2.SafetensorDataProto) -> services_pb2.SafetensorDataProto:
        # TODO agg logic fix
        request_data = safetensors.torch.load(request.data)
        client_id = request.client_id
        client_iteration = request.iteration
        logger.info(f"Got aggregation query from {client_id}")
        self._agg_results[client_iteration][client_id] = request_data['tensor']
        while len(self._agg_results[client_iteration]) != self.world_size:
            continue

        returned_tensor = torch.sum(torch.stack(list(self._agg_results[client_iteration].values())), axis=0)
        self._agg_results[client_iteration] = dict()
        logger.info(f"All queries collected, returning aggregated result")
        return services_pb2.SafetensorDataProto(
            data=safetensors.torch.save(tensors={'tensor': returned_tensor}),
            client_id=client_id,
            iteration=client_iteration,
        )


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


def serve():
    try:
        port = "50051"
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
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

# Пока они не начнут пинги, потом когда они все ready to start, сервер берет их последние запросы на подключение (пинги) и
# в ответ на эти пинги присылает первоначальную конфигурацию то есть есть некоторый словарь содержащий их последние реквесты и перед тем как эти реве
