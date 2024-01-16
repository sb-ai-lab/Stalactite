# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc

import stalactite.communications.grpc_utils.generated_code.arbiter_pb2 as arbiter__pb2


class ArbiterStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.DecodeMessage = channel.unary_unary(
                '/Arbiter/DecodeMessage',
                request_serializer=arbiter__pb2.DataMessage.SerializeToString,
                response_deserializer=arbiter__pb2.DataMessage.FromString,
                )
        self.GetPublicKey = channel.unary_unary(
                '/Arbiter/GetPublicKey',
                request_serializer=arbiter__pb2.RequestResponse.SerializeToString,
                response_deserializer=arbiter__pb2.PublicContext.FromString,
                )
        self.GenerateKeys = channel.unary_unary(
                '/Arbiter/GenerateKeys',
                request_serializer=arbiter__pb2.RequestResponse.SerializeToString,
                response_deserializer=arbiter__pb2.RequestResponse.FromString,
                )


class ArbiterServicer(object):
    """Missing associated documentation comment in .proto file."""

    def DecodeMessage(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def GetPublicKey(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def GenerateKeys(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_ArbiterServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'DecodeMessage': grpc.unary_unary_rpc_method_handler(
                    servicer.DecodeMessage,
                    request_deserializer=arbiter__pb2.DataMessage.FromString,
                    response_serializer=arbiter__pb2.DataMessage.SerializeToString,
            ),
            'GetPublicKey': grpc.unary_unary_rpc_method_handler(
                    servicer.GetPublicKey,
                    request_deserializer=arbiter__pb2.RequestResponse.FromString,
                    response_serializer=arbiter__pb2.PublicContext.SerializeToString,
            ),
            'GenerateKeys': grpc.unary_unary_rpc_method_handler(
                    servicer.GenerateKeys,
                    request_deserializer=arbiter__pb2.RequestResponse.FromString,
                    response_serializer=arbiter__pb2.RequestResponse.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'Arbiter', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class Arbiter(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def DecodeMessage(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/Arbiter/DecodeMessage',
            arbiter__pb2.DataMessage.SerializeToString,
            arbiter__pb2.DataMessage.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def GetPublicKey(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/Arbiter/GetPublicKey',
            arbiter__pb2.RequestResponse.SerializeToString,
            arbiter__pb2.PublicContext.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def GenerateKeys(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/Arbiter/GenerateKeys',
            arbiter__pb2.RequestResponse.SerializeToString,
            arbiter__pb2.RequestResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)