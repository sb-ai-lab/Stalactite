# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: services.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x0eservices.proto\"\x95\x01\n\x13SafetensorDataProto\x12\x0c\n\x04\x64\x61ta\x18\x01 \x01(\x0c\x12\x11\n\titeration\x18\x02 \x01(\x05\x12\x11\n\tclient_id\x18\x03 \x01(\t\x12\x12\n\x05\x62\x61tch\x18\x04 \x01(\x05H\x00\x88\x01\x01\x12\x1a\n\rtotal_batches\x18\x05 \x01(\x05H\x01\x88\x01\x01\x42\x08\n\x06_batchB\x10\n\x0e_total_batches\"F\n\x04Ping\x12\x0c\n\x04\x64\x61ta\x18\x01 \x01(\t\x12\x1c\n\x0f\x61\x64\x64itional_info\x18\x02 \x01(\tH\x00\x88\x01\x01\x42\x12\n\x10_additional_info\"\xe2\x03\n\x0bTensorProto\x12\x0c\n\x04\x64ims\x18\x01 \x03(\x03\x12(\n\tdata_type\x18\x02 \x01(\x0e\x32\x15.TensorProto.DataType\x12\x12\n\nfloat_data\x18\x03 \x03(\x01\x12\x12\n\nint32_data\x18\x04 \x03(\x05\x12\x11\n\tbyte_data\x18\x05 \x01(\x0c\x12\x13\n\x0bstring_data\x18\x06 \x03(\x0c\x12\x13\n\x0b\x64ouble_data\x18\x07 \x03(\x01\x12\x12\n\nint64_data\x18\x08 \x03(\x03\x12\x0e\n\x06status\x18\t \x01(\x05\x12\x11\n\titeration\x18\n \x01(\x05\x12\x11\n\tclient_id\x18\x0b \x01(\t\x12\x12\n\x05\x62\x61tch\x18\x0c \x01(\x05H\x00\x88\x01\x01\x12\x1a\n\rtotal_batches\x18\r \x01(\x05H\x01\x88\x01\x01\"\x9f\x01\n\x08\x44\x61taType\x12\r\n\tUNDEFINED\x10\x00\x12\t\n\x05\x46LOAT\x10\x01\x12\t\n\x05INT32\x10\x02\x12\x08\n\x04\x42YTE\x10\x03\x12\n\n\x06STRING\x10\x04\x12\x08\n\x04\x42OOL\x10\x05\x12\t\n\x05UINT8\x10\x06\x12\x08\n\x04INT8\x10\x07\x12\n\n\x06UINT16\x10\x08\x12\t\n\x05INT16\x10\t\x12\t\n\x05INT64\x10\n\x12\x0b\n\x07\x46LOAT16\x10\x0c\x12\n\n\x06\x44OUBLE\x10\rB\x08\n\x06_batchB\x10\n\x0e_total_batches2\xea\x03\n\x0c\x43ommunicator\x12\x1e\n\x08PingPong\x12\x05.Ping\x1a\x05.Ping\"\x00(\x01\x30\x01\x12O\n\x1f\x45xchangeBinarizedDataUnaryUnary\x12\x14.SafetensorDataProto\x1a\x14.SafetensorDataProto\"\x00\x12;\n\x1b\x45xchangeNumpyDataUnaryUnary\x12\x0c.TensorProto\x1a\x0c.TensorProto\"\x00\x12R\n ExchangeBinarizedDataStreamUnary\x12\x14.SafetensorDataProto\x1a\x14.SafetensorDataProto\"\x00(\x01\x12>\n\x1c\x45xchangeNumpyDataStreamUnary\x12\x0c.TensorProto\x1a\x0c.TensorProto\"\x00(\x01\x12U\n!ExchangeBinarizedDataStreamStream\x12\x14.SafetensorDataProto\x1a\x14.SafetensorDataProto\"\x00(\x01\x30\x01\x12\x41\n\x1d\x45xchangeNumpyDataStreamStream\x12\x0c.TensorProto\x1a\x0c.TensorProto\"\x00(\x01\x30\x01\x62\x06proto3')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'services_pb2', _globals)
if _descriptor._USE_C_DESCRIPTORS == False:
  DESCRIPTOR._options = None
  _globals['_SAFETENSORDATAPROTO']._serialized_start=19
  _globals['_SAFETENSORDATAPROTO']._serialized_end=168
  _globals['_PING']._serialized_start=170
  _globals['_PING']._serialized_end=240
  _globals['_TENSORPROTO']._serialized_start=243
  _globals['_TENSORPROTO']._serialized_end=725
  _globals['_TENSORPROTO_DATATYPE']._serialized_start=538
  _globals['_TENSORPROTO_DATATYPE']._serialized_end=697
  _globals['_COMMUNICATOR']._serialized_start=728
  _globals['_COMMUNICATOR']._serialized_end=1218
# @@protoc_insertion_point(module_scope)
