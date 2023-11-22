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



# exchange data method for bytes in grpc proto
# message ping?
# message data (any)
# message training conf?


# при инициализаци стартует сервер либо клиент (отдельные запуски)
# Менеджер Сервера стартует, публикует порт, ждет
# Сервер стаб реализует методы аггрегации, сбора данных, чего то там еще
# Клиенты подключаются (пингуют, сервер добавляет каждого клиента в словарь в котором еще канал с этим клиентом, при отключении
# Сервер инициирует запуск кода который передан

# Просто сервер с коннекшнами и стартуют обмены пока только мок


# Агент запусается (питоновский процесс, там етсь grpc тока и хендлер в котором сразу есть объект партимембер, в него прилетают вызовы grpc с методом, по которому будут вызываться методы )
# А внутри патри мембера модель параллельно с grpc точкой поток с сообщениями HB

# На сервере та же точка (процесс) с базовыми методами вызовы методов парти мастера и отдельно HB

# потом будут тесты
# юнит тесты без агента процессов
# агенты докер контейнеры в одной машине с обменом между ними
# end-to-end тесты с деплоем на боевую среду

# на следующем шаге увеличение объема, а потом среда (было локаьлно со скоростью сериализации) потом сбер + янд облако + наш кластер