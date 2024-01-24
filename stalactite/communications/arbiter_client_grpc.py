import logging
from typing import Optional, Union

import grpc
import tenseal as ts
import torch

from stalactite.communications.grpc_utils.generated_code import arbiter_pb2_grpc, arbiter_pb2
from stalactite.communications.grpc_utils.utils import ArbiterServerError, load_data

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
sh = logging.StreamHandler()
sh.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(sh)


class GRpcArbiterCommunicator:
    """ gRPC Arbiter communicator class. """

    def __init__(
            self,
            master_id: str,
            arbiter_host: str = '0.0.0.0',
            arbiter_port: str = '50052',
            grpc_operations_timeout: float = 300.,
            max_message_size: int = -1,
    ):
        """
        Initialize GRpcArbiterCommunicator class.

        :param master_id: ID of the VFL experiment master
        :param arbiter_host: Host of the gRPC server
        :param arbiter_port: Port of the gRPC server
        :param grpc_operations_timeout: Timeout of the gRPC operations
        :param max_message_size: Maximum message length that the gRPC channel can send or receive. -1 means unlimited
        """
        self.master_id = master_id
        self.arbiter_host = arbiter_host
        self.arbiter_port = arbiter_port
        self.grpc_operations_timeout = grpc_operations_timeout
        self.max_message_size = max_message_size

        self._tenseal_context: Optional[ts.Context] = None
        self._grpc_channel: Optional[grpc.Channel] = None
        self._is_initialized = False

    @property
    def is_initialized(self) -> bool:
        """ Whether the arbiter was initialized. """
        return self._is_initialized

    def initialize_arbiter(self):
        """
        Initialize arbiter client.
        Create channel, generate keys on arbiter server.
        """
        self._grpc_channel = grpc.insecure_channel(
            f"{self.arbiter_host}:{self.arbiter_port}",
            options=[
                ('grpc.max_send_message_length', self.max_message_size),
                ('grpc.max_receive_message_length', self.max_message_size),
            ]
        )
        self._stub = arbiter_pb2_grpc.ArbiterStub(self._grpc_channel)
        logger.info(f"Initialized arbiter client: ({self.arbiter_host}:{self.arbiter_port})")

        self._generate_keys()
        self._is_initialized = True

    def _generate_keys(self):
        """ Call the keys generation endpoint if arbiter has not been initialized yet. """
        if not self.is_initialized:
            response = self._stub.GenerateKeys(
                arbiter_pb2.RequestResponse(master_id=self.master_id, request_response=True),
                timeout=self.grpc_operations_timeout,
                wait_for_ready=True,
            )
            if error_msg := response.error:
                raise ArbiterServerError(message=error_msg)
        else:
            raise RuntimeError('Arbiter has been already initialized. Generation of the new keys will lead to errors.')

    def _get_public_key(self) -> ts.Context:
        """ Call public key getter endpoint, add tenseal public context. """
        try:
            response = self._stub.GetPublicKey(
                arbiter_pb2.RequestResponse(master_id=self.master_id, request_response=True),
                timeout=self.grpc_operations_timeout,
                wait_for_ready=True,
            )
            if error_msg := response.error:
                raise ArbiterServerError(message=error_msg)
            if (serialized_context := response.pubkey) is None:
                raise RuntimeError('Arbiter server did not return any key.')
            context = ts.context_from(serialized_context)
            assert context.is_public(), 'Returned context is not public. Cannot proceed with private context.'
            self._tenseal_context = context
            return context
        except Exception as exc:
            raise exc

    @property
    def public_key(self) -> ts.Context:
        """ Get public tenseal context, call server if run for the first time. """
        if self._tenseal_context is not None:
            return self._tenseal_context
        else:
            return self._get_public_key()

    def decrypt_data(
            self, encrypted_data: Union[ts.BFVTensor, ts.BFVTensor, ts.CKKSVector, ts.CKKSTensor]
    ) -> torch.Tensor:
        """
        Decrypt data with private key on server.

        :param encrypted_data: Data encrypted with a public tenseal context to decrypt
        """
        try:
            response = self._stub.DecodeMessage(
                arbiter_pb2.DataMessage(
                    master_id=self.master_id,
                    encrypted_data=encrypted_data.serialize(),
                    data_shape=list(encrypted_data.shape),
                ),
                timeout=self.grpc_operations_timeout,
                wait_for_ready=True,
            )
            if error_msg := response.error:
                raise ArbiterServerError(message=error_msg)
            return load_data(response.decrypted_tensor)
        except Exception as exc:
            raise exc
