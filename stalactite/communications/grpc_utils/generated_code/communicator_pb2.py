# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: communicator.proto
# Protobuf Python Version: 4.25.0
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder

# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x12\x63ommunicator.proto\"C\n\x08SendTime\x12\x0f\n\x07task_id\x18\x01 \x01(\t\x12\x13\n\x0bmethod_name\x18\x02 \x01(\t\x12\x11\n\tsend_time\x18\x03 \x01(\x02\"I\n\x02HB\x12\x12\n\nagent_name\x18\x01 \x01(\t\x12\x0e\n\x06status\x18\x02 \x01(\t\x12\x1f\n\x0csend_timings\x18\x03 \x03(\x0b\x32\t.SendTime\"\xde\x04\n\x0bMainMessage\x12\x14\n\x0cmessage_type\x18\x01 \x01(\t\x12\x1b\n\x0erequire_answer\x18\x02 \x01(\x08H\x00\x88\x01\x01\x12\x13\n\x06status\x18\x03 \x01(\tH\x01\x88\x01\x01\x12\x14\n\x07task_id\x18\x04 \x01(\tH\x02\x88\x01\x01\x12\x16\n\tparent_id\x18\x05 \x01(\tH\x03\x88\x01\x01\x12\x15\n\x08\x66rom_uid\x18\x06 \x01(\tH\x04\x88\x01\x01\x12\x18\n\x0bmethod_name\x18\x07 \x01(\tH\x05\x88\x01\x01\x12\x1f\n\x12parent_method_name\x18\x08 \x01(\tH\x06\x88\x01\x01\x12\'\n\x1aparent_task_execution_time\x18\t \x01(\x02H\x07\x88\x01\x01\x12\x35\n\rtensor_kwargs\x18\n \x03(\x0b\x32\x1e.MainMessage.TensorKwargsEntry\x12\x33\n\x0cother_kwargs\x18\x0b \x03(\x0b\x32\x1d.MainMessage.OtherKwargsEntry\x1a\x33\n\x11TensorKwargsEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\r\n\x05value\x18\x02 \x01(\x0c:\x02\x38\x01\x1a\x32\n\x10OtherKwargsEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\r\n\x05value\x18\x02 \x01(\x0c:\x02\x38\x01\x42\x11\n\x0f_require_answerB\t\n\x07_statusB\n\n\x08_task_idB\x0c\n\n_parent_idB\x0b\n\t_from_uidB\x0e\n\x0c_method_nameB\x15\n\x13_parent_method_nameB\x1d\n\x1b_parent_task_execution_time2\x8c\x01\n\x0c\x43ommunicator\x12\x1b\n\tHeartbeat\x12\x03.HB\x1a\x03.HB\"\x00(\x01\x30\x01\x12\x30\n\x0c\x42idiExchange\x12\x0c.MainMessage\x1a\x0c.MainMessage\"\x00(\x01\x30\x01\x12-\n\rUnaryExchange\x12\x0c.MainMessage\x1a\x0c.MainMessage\"\x00\x62\x06proto3')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'communicator_pb2', _globals)
if _descriptor._USE_C_DESCRIPTORS == False:
  DESCRIPTOR._options = None
  _globals['_MAINMESSAGE_TENSORKWARGSENTRY']._options = None
  _globals['_MAINMESSAGE_TENSORKWARGSENTRY']._serialized_options = b'8\001'
  _globals['_MAINMESSAGE_OTHERKWARGSENTRY']._options = None
  _globals['_MAINMESSAGE_OTHERKWARGSENTRY']._serialized_options = b'8\001'
  _globals['_SENDTIME']._serialized_start=22
  _globals['_SENDTIME']._serialized_end=89
  _globals['_HB']._serialized_start=91
  _globals['_HB']._serialized_end=164
  _globals['_MAINMESSAGE']._serialized_start=167
  _globals['_MAINMESSAGE']._serialized_end=773
  _globals['_MAINMESSAGE_TENSORKWARGSENTRY']._serialized_start=531
  _globals['_MAINMESSAGE_TENSORKWARGSENTRY']._serialized_end=582
  _globals['_MAINMESSAGE_OTHERKWARGSENTRY']._serialized_start=584
  _globals['_MAINMESSAGE_OTHERKWARGSENTRY']._serialized_end=634
  _globals['_COMMUNICATOR']._serialized_start=776
  _globals['_COMMUNICATOR']._serialized_end=916
# @@protoc_insertion_point(module_scope)
