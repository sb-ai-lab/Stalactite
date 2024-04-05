from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Mapping as _Mapping, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class IsReady(_message.Message):
    __slots__ = ("sender_id", "ready", "role")
    SENDER_ID_FIELD_NUMBER: _ClassVar[int]
    READY_FIELD_NUMBER: _ClassVar[int]
    ROLE_FIELD_NUMBER: _ClassVar[int]
    sender_id: str
    ready: bool
    role: str
    def __init__(self, sender_id: _Optional[str] = ..., ready: bool = ..., role: _Optional[str] = ...) -> None: ...

class MainArbiterMessage(_message.Message):
    __slots__ = ("sender_id", "task_id", "method_name", "tensor_kwargs", "other_kwargs", "get_response_timeout")
    class TensorKwargsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: bytes
        def __init__(self, key: _Optional[str] = ..., value: _Optional[bytes] = ...) -> None: ...
    class OtherKwargsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: bytes
        def __init__(self, key: _Optional[str] = ..., value: _Optional[bytes] = ...) -> None: ...
    SENDER_ID_FIELD_NUMBER: _ClassVar[int]
    TASK_ID_FIELD_NUMBER: _ClassVar[int]
    METHOD_NAME_FIELD_NUMBER: _ClassVar[int]
    TENSOR_KWARGS_FIELD_NUMBER: _ClassVar[int]
    OTHER_KWARGS_FIELD_NUMBER: _ClassVar[int]
    GET_RESPONSE_TIMEOUT_FIELD_NUMBER: _ClassVar[int]
    sender_id: str
    task_id: str
    method_name: str
    tensor_kwargs: _containers.ScalarMap[str, bytes]
    other_kwargs: _containers.ScalarMap[str, bytes]
    get_response_timeout: float
    def __init__(self, sender_id: _Optional[str] = ..., task_id: _Optional[str] = ..., method_name: _Optional[str] = ..., tensor_kwargs: _Optional[_Mapping[str, bytes]] = ..., other_kwargs: _Optional[_Mapping[str, bytes]] = ..., get_response_timeout: _Optional[float] = ...) -> None: ...
