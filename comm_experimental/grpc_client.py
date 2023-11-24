import logging
import time
from typing import Iterator
from threading import Thread
import uuid

import grpc
import torch
import safetensors.torch

from gen_code import services_pb2, services_pb2_grpc
from helpers import PingResponse, ClientStatus

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
            # client_status_check_interval: float = 1.,
            unary_unary_operations_timeout: float = 5000.,
    ):
        self.ping_interval = ping_interval
        # self.client_status_check_interval = client_status_check_interval
        self.unary_unary_operations_timeout = unary_unary_operations_timeout

        self._uuid = uuid.uuid4()
        self._grpc_channel = grpc.insecure_channel(f"{server_host}:{server_port}")
        self._stub = services_pb2_grpc.CommunicatorStub(self._grpc_channel)

        self._status = ClientStatus.waiting
        self._iteration = 1
        self._pings_thread = None

        logger.info(f"Initialized {self._uuid} client: ({server_host}:{server_port})")

    @property
    def iteration(self):
        return self._iteration

    @property
    def client_name(self):
        return str(self._uuid)

    @property
    def ping_message(self):
        return services_pb2.Ping(data=self.client_name)

    def ping_messages_generator(self):
        while True:
            time.sleep(self.ping_interval)
            yield self.ping_message

    def process_pongs(self, server_response_iterator: Iterator[services_pb2.Ping]):
        try:
            for response in server_response_iterator:
                logger.debug(f"Got pong from server: {response.data}")
                message = response.data
                if message == PingResponse.waiting_for_other_connections:
                    self._status = ClientStatus.waiting
                elif message == PingResponse.all_ready:
                    logger.info("Got `all ready` pong from server")
                    self._status = ClientStatus.active
        except Exception as exc:
            print(exc)

    def _get_future(self, ):
        future = self._stub.ExchangeBinarizedDataUnaryUnary.future(
            services_pb2.SafetensorDataProto(
                data=safetensors.torch.save(
                    tensors={'tensor': torch.tensor([1, 2], dtype=torch.float32)},
                ),
                client_id=self.client_name,
                iteration=self.iteration,
            ),
            timeout=self.unary_unary_operations_timeout,
        )
        return future

    def run(self):
        while True:
            if self._status == ClientStatus.waiting:
                continue
            elif self._status == ClientStatus.active:
                # TODO RPC calls here: Реализовать через добавление в очередь тасок (в тч таска finish)
                future = self._get_future()
                data = future.result()
                logger.info(f"Got aggregation result from server: {safetensors.torch.load(data.data)}")
                self._status = ClientStatus.finished
            else:
                logger.info(f"Client {self.client_name} finished run. Terminating...")
                break

    def start(self):
        try:
            # Start ping-pong thread
            # print('ininin')
            logger.info("Starting ping-pong with the server")
            pingpong_responses = self._stub.PingPong(
                self.ping_messages_generator(),
                wait_for_ready=True,
            )
            self._pings_thread = Thread(target=self.process_pongs, args=(pingpong_responses,))
            self._pings_thread.daemon = True
            self._pings_thread.start()
            self.run()

            self._close()
        except KeyboardInterrupt:
            self._close()

    def _close(self):
        self._grpc_channel.close()
        # self._pings_thread.join(timeout=0)
        # logger.debug("Joined ping-pong thread")



if __name__ == '__main__':
    # logging.basicConfig()
    client = GRpcClient(server_host='0.0.0.0', server_port=50051)
    client.start()
