import asyncio
from concurrent import futures
import logging
import os.path
from typing import Optional

import click
import grpc
import tenseal as ts
import torch

from stalactite.communications.grpc_utils.generated_code import arbiter_pb2, arbiter_pb2_grpc
from stalactite.communications.grpc_utils.utils import save_data
from stalactite.configs import VFLConfig

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
sh = logging.StreamHandler()
sh.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(sh)


class GRpcArbiterServicer(arbiter_pb2_grpc.ArbiterServicer):
    def __init__(
            self,
            ts_algorithm: ts.SCHEME_TYPE = ts.SCHEME_TYPE.CKKS,
            ts_poly_modulus_degree: int = 8192,
            ts_coeff_mod_bit_sizes: list[int] = [60, 40, 40, 60],
            ts_global_scale: int = 2 ** 20,
            ts_plain_modulus: Optional[int] = None,
            ts_generate_galois_keys: bool = True,
            ts_generate_relin_keys: bool = True,
            ts_context_path: Optional[str] = None,
    ):
        """
        Initialize GRpcArbiterServicer class.

        :param ts_algorithm: Tenseal param. One of ts.SCHEME_TYPE.CKKS (used for real values) 
               | ts.SCHEME_TYPE.BFV (used for integers)
        :param ts_poly_modulus_degree: Tenseal param. The degree of the polynomial modulus, must be a power of two
        :param ts_coeff_mod_bit_sizes: Tenseal param. List of bit size for each coefficient modulus.
               Can be an empty list for BFV, a default value will be given
        :param ts_global_scale: Tenseal param. Scale of ciphertext to use
        :param ts_generate_galois_keys: Whether to generate galois keys 
               (galois keys are required to do ciphertext rotations)
        :param ts_generate_relin_keys: Whether to generate relinearization keys (needed for encrypted multiplications)
        :param ts_context_path: Path to the file containing serialized private tenseal context
        """
        self.ts_algorithm = ts_algorithm
        self.ts_poly_modulus_degree = ts_poly_modulus_degree
        self.ts_coeff_mod_bit_sizes = ts_coeff_mod_bit_sizes
        self.ts_global_scale = ts_global_scale
        self.ts_plain_modulus = ts_plain_modulus
        self.ts_generate_galois_keys = ts_generate_galois_keys
        self.ts_generate_relin_keys = ts_generate_relin_keys
        self.ts_context_path = ts_context_path

        if self.ts_plain_modulus is not None and self.ts_algorithm == ts.SCHEME_TYPE.CKKS:
            raise ValueError('ts_plain_modulus: Should not be passed when the scheme is CKKS.')

        self._tenseal_secret_context: dict[str, Optional[ts.Context]] = dict()
        self._tenseal_public_context: dict[str, Optional[ts.Context]] = dict()

    def tenseal_secret_context(self, master_id: str) -> Optional[ts.Context]:
        """ Return dictionary of secret contexts. """
        return self._tenseal_secret_context.get(master_id)

    def tenseal_public_context(self, master_id) -> Optional[ts.Context]:
        """ Return dictionary of public contexts. """
        return self._tenseal_public_context.get(master_id)

    def _load_ts_context(self) -> ts.Context:
        """ Load tenseal context from `ts_context_path`. """
        logger.info(f'Loading Tenseal context from {self.ts_context_path}')
        if not os.path.exists(self.ts_context_path):
            raise FileExistsError(f'File not exist: {self.ts_context_path}.')
        with open(self.ts_context_path, 'rb') as f:
            serialized_context = f.read()
        context = ts.context_from(serialized_context)
        return context

    def _generate_keys(self, master_id: str):
        """
        Generate or load (if `ts_context_path` is provided) private and public contexts.

        :param master_id: id of the VFL master
        """
        if self.ts_context_path is None:
            context = ts.context(
                self.ts_algorithm,
                poly_modulus_degree=self.ts_poly_modulus_degree,
                coeff_mod_bit_sizes=self.ts_coeff_mod_bit_sizes,
            )
            if self.ts_generate_galois_keys:
                context.generate_galois_keys()
            if self.ts_generate_relin_keys:
                context.generate_relin_keys()
            context.global_scale = self.ts_global_scale
        else:
            try:
                context = self._load_ts_context()
            except FileExistsError as exc:
                raise exc
            if not context.is_private():
                raise ValueError('`ts_context_path` contains public context. Private context must be used.')
        self._tenseal_secret_context[master_id] = context
        context_pub = context.copy()
        context_pub.make_context_public()
        self._tenseal_public_context[master_id] = context_pub

    async def GenerateKeys(
            self,
            request: arbiter_pb2.RequestResponse,
            context: grpc.aio.ServicerContext,
    ) -> arbiter_pb2.RequestResponse:
        """ Generate keys for the VFL experiment. """
        if request.request_response:
            if self.tenseal_public_context(master_id=request.master_id) is not None:
                # request_response=False because already the keys has been already generated
                response = arbiter_pb2.RequestResponse(request_response=False, master_id=request.master_id)
            else:
                try:
                    self._generate_keys(master_id=request.master_id)
                    response = arbiter_pb2.RequestResponse(request_response=True, master_id=request.master_id)
                except Exception as exc:
                    response = arbiter_pb2.RequestResponse(
                        request_response=False, error=str(exc), master_id=request.master_id
                    )
            return response

    async def GetPublicKey(
            self,
            request: arbiter_pb2.RequestResponse,
            context: grpc.aio.ServicerContext,
    ) -> arbiter_pb2.PublicContext:
        """ Return public key to the VFL agent. """
        try:
            context = self.tenseal_public_context(master_id=request.master_id)
            if context is None:
                return arbiter_pb2.PublicContext(
                    master_id=request.master_id,
                    error='No public context generated, call `GenerateKeys` endpoint first.'
                )
            if not context.is_public():
                raise RuntimeError('Context generation failed, public context has not been created.')
            return arbiter_pb2.PublicContext(master_id=request.master_id, pubkey=context.serialize())
        except Exception as exc:
            return arbiter_pb2.PublicContext(master_id=request.master_id, error=str(exc))

    async def DecodeMessage(
            self,
            request: arbiter_pb2.DataMessage,
            context: grpc.aio.ServicerContext,
    ) -> arbiter_pb2.DataMessage:
        """ Decrypt with a private key encrypted with a public key tensor. """
        if self.tenseal_secret_context(master_id=request.master_id) is None:
            return arbiter_pb2.DataMessage(
                master_id=request.master_id, error='No context generated, call `GenerateKeys` endpoint first.'
            )
        if request.encrypted_data is None:
            return arbiter_pb2.DataMessage(master_id=request.master_id, error='No data sent.')
        try:
            data_shape = request.data_shape
            if len(data_shape) == 1:
                deserialization_func = ts.ckks_vector_from
            elif len(data_shape) > 1:
                deserialization_func = ts.ckks_tensor_from
            else:
                raise RuntimeError('Did not get a shape of the data to decrypt.')
            encrypted_data = deserialization_func(
                self.tenseal_secret_context(master_id=request.master_id),
                request.encrypted_data
            )
            data = encrypted_data.decrypt()
            if len(data_shape) == 1:
                return arbiter_pb2.DataMessage(
                    master_id=request.master_id,
                    decrypted_tensor=save_data(torch.tensor(data))
                )
            else:
                data_to_sent = torch.tensor(data.raw).reshape(data.shape)
                return arbiter_pb2.DataMessage(master_id=request.master_id, decrypted_tensor=save_data(data_to_sent))
        except Exception as exc:
            return arbiter_pb2.DataMessage(master_id=request.master_id, error=str(exc))


async def serve(config_path: str):
    """ Initialize and start gRPC arbiter server. """
    config = VFLConfig.load_and_validate(config_path)
    arbiter_server = grpc.aio.server(
        futures.ThreadPoolExecutor(max_workers=config.grpc_arbiter.server_threadpool_max_workers),
        options=[
            ('grpc.max_send_message_length', config.grpc_arbiter.max_message_size),
            ('grpc.max_receive_message_length', config.grpc_arbiter.max_message_size)
        ]
    )
    arbiter_servicer = GRpcArbiterServicer(
        ts_algorithm=config.grpc_arbiter.ts_algorithm,
        ts_poly_modulus_degree=config.grpc_arbiter.ts_poly_modulus_degree,
        ts_coeff_mod_bit_sizes=config.grpc_arbiter.ts_coeff_mod_bit_sizes,
        ts_global_scale=2 ** config.grpc_arbiter.ts_global_scale_pow,
        ts_plain_modulus=config.grpc_arbiter.ts_plain_modulus,
        ts_generate_galois_keys=config.grpc_arbiter.ts_generate_galois_keys,
        ts_generate_relin_keys=config.grpc_arbiter.ts_generate_relin_keys,
        ts_context_path=config.grpc_arbiter.ts_context_path,
    )
    arbiter_pb2_grpc.add_ArbiterServicer_to_server(arbiter_servicer, arbiter_server)
    arbiter_server.add_insecure_port(f"{config.grpc_arbiter.host}:{config.grpc_arbiter.port}")  # TODO SSL goes here
    logger.info(f'Starting arbiter at {config.grpc_arbiter.host}:{config.grpc_arbiter.port}')
    await arbiter_server.start()
    await arbiter_server.wait_for_termination()


@click.command()
@click.option('--config-path', type=str, required=True)
def main(config_path):
    try:
        asyncio.get_event_loop().run_until_complete(serve(config_path=config_path))
    except KeyboardInterrupt:
        logger.info('Terminated')


if __name__ == '__main__':
    main()
