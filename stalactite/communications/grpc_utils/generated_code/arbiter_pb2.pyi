from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class RequestResponse(_message.Message):
    __slots__ = ("master_id", "request_response", "error")
    MASTER_ID_FIELD_NUMBER: _ClassVar[int]
    REQUEST_RESPONSE_FIELD_NUMBER: _ClassVar[int]
    ERROR_FIELD_NUMBER: _ClassVar[int]
    master_id: str
    request_response: bool
    error: str
    def __init__(self, master_id: _Optional[str] = ..., request_response: bool = ..., error: _Optional[str] = ...) -> None: ...

class PublicContext(_message.Message):
    __slots__ = ("master_id", "pubkey", "error")
    MASTER_ID_FIELD_NUMBER: _ClassVar[int]
    PUBKEY_FIELD_NUMBER: _ClassVar[int]
    ERROR_FIELD_NUMBER: _ClassVar[int]
    master_id: str
    pubkey: bytes
    error: str
    def __init__(self, master_id: _Optional[str] = ..., pubkey: _Optional[bytes] = ..., error: _Optional[str] = ...) -> None: ...

class DataMessage(_message.Message):
    __slots__ = ("master_id", "encrypted_data", "decrypted_tensor", "error", "data_shape")
    MASTER_ID_FIELD_NUMBER: _ClassVar[int]
    ENCRYPTED_DATA_FIELD_NUMBER: _ClassVar[int]
    DECRYPTED_TENSOR_FIELD_NUMBER: _ClassVar[int]
    ERROR_FIELD_NUMBER: _ClassVar[int]
    DATA_SHAPE_FIELD_NUMBER: _ClassVar[int]
    master_id: str
    encrypted_data: bytes
    decrypted_tensor: bytes
    error: str
    data_shape: _containers.RepeatedScalarFieldContainer[int]
    def __init__(self, master_id: _Optional[str] = ..., encrypted_data: _Optional[bytes] = ..., decrypted_tensor: _Optional[bytes] = ..., error: _Optional[str] = ..., data_shape: _Optional[_Iterable[int]] = ...) -> None: ...
