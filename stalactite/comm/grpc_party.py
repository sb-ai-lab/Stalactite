from concurrent import futures

import grpc

from stalactite.comm.abc_party import Party


class GRPCServerManager:
    """ Serving / termination / handling connections list """
    def __init__(self, host: str, port: int, world_size: int, server_workers: int = 10):
        self.world_size = world_size


        self.grpc_server = grpc.server(futures.ThreadPoolExecutor(max_workers=server_workers))
        # processor_pb2_grpc.add_ProcessorServicer_to_server(ProcessorServicer(), self.grpc_server) # TODO add
        self.grpc_server.add_insecure_port(f"{host}:{port}")

        # self.grpc_server.add_generic_rpc_handlers([grpc.stream_stream_rpc_method_handler(), ])
        # Handle (de-)serialization
        self._clients = []

    @property
    def list_all_connections(self):
        return self._clients

    @property
    def if_clients_connected(self):
        # For server only
        return self.world_size == len(self.list_all_connections)

    def start(self):
        self.grpc_server.start()
        # print("Started gRPC server: 0.0.0.0:50051")
        # server loop to keep the process running
        self.grpc_server.wait_for_termination()


class GRPCClient:
    def __init__(self, host: str, port: int):
        # Insecure for now TODO
        self.channel = grpc.insecure_channel(f"{host}:{port}")

        self.stub = None


    def start(self):
        # self.stub = route_guide_pb2_grpc.RouteGuideStub(self.channel)
        raise NotImplementedError


class GRPCParty(Party):
    def __init__(self, world_size: int, rank: int):
        self._world_size = world_size
        self._rank = rank



    @property
    def world_size(self) -> int:
        return self._world_size

    def send(self, method_name: str, mass_kwargs: dict[str, list], **kwargs):
        """

        :param method_name:
        :param mass_kwargs:
        :param kwargs:
        :return:
        """
        raise NotImplementedError
