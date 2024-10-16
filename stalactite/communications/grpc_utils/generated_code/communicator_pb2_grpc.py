# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc

import stalactite.communications.grpc_utils.generated_code.communicator_pb2 as communicator__pb2


class CommunicatorStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.Heartbeat = channel.stream_stream(
                '/Communicator/Heartbeat',
                request_serializer=communicator__pb2.HB.SerializeToString,
                response_deserializer=communicator__pb2.HB.FromString,
                )
        self.SendToMaster = channel.unary_unary(
                '/Communicator/SendToMaster',
                request_serializer=communicator__pb2.MainMessage.SerializeToString,
                response_deserializer=communicator__pb2.MainMessage.FromString,
                )
        self.RecvFromMaster = channel.unary_unary(
                '/Communicator/RecvFromMaster',
                request_serializer=communicator__pb2.MainMessage.SerializeToString,
                response_deserializer=communicator__pb2.MainMessage.FromString,
                )


class CommunicatorServicer(object):
    """Missing associated documentation comment in .proto file."""

    def Heartbeat(self, request_iterator, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def SendToMaster(self, request, context):
        """rpc BidiExchange(stream MainMessage) returns (stream MainMessage) {}
        rpc UnaryExchange(MainMessage) returns (MainMessage) {}
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def RecvFromMaster(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_CommunicatorServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'Heartbeat': grpc.stream_stream_rpc_method_handler(
                    servicer.Heartbeat,
                    request_deserializer=communicator__pb2.HB.FromString,
                    response_serializer=communicator__pb2.HB.SerializeToString,
            ),
            'SendToMaster': grpc.unary_unary_rpc_method_handler(
                    servicer.SendToMaster,
                    request_deserializer=communicator__pb2.MainMessage.FromString,
                    response_serializer=communicator__pb2.MainMessage.SerializeToString,
            ),
            'RecvFromMaster': grpc.unary_unary_rpc_method_handler(
                    servicer.RecvFromMaster,
                    request_deserializer=communicator__pb2.MainMessage.FromString,
                    response_serializer=communicator__pb2.MainMessage.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'Communicator', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class Communicator(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def Heartbeat(request_iterator,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.stream_stream(request_iterator, target, '/Communicator/Heartbeat',
            communicator__pb2.HB.SerializeToString,
            communicator__pb2.HB.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def SendToMaster(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/Communicator/SendToMaster',
            communicator__pb2.MainMessage.SerializeToString,
            communicator__pb2.MainMessage.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def RecvFromMaster(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/Communicator/RecvFromMaster',
            communicator__pb2.MainMessage.SerializeToString,
            communicator__pb2.MainMessage.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)
