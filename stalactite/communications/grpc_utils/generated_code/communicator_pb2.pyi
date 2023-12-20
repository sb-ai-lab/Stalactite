from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Mapping as _Mapping, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class HB(_message.Message):
    __slots__ = ["agent_name", "status"]
    AGENT_NAME_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    agent_name: str
    status: str
    def __init__(self, agent_name: _Optional[str] = ..., status: _Optional[str] = ...) -> None: ...

class MainMessage(_message.Message):
    __slots__ = ["message_type", "require_answer", "status", "task_id", "parent_id", "from_uid", "method_name", "numeric_kwargs", "str_kwargs", "bytes_kwargs"]
    class NumericKwargsEntry(_message.Message):
        __slots__ = ["key", "value"]
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: float
        def __init__(self, key: _Optional[str] = ..., value: _Optional[float] = ...) -> None: ...
    class StrKwargsEntry(_message.Message):
        __slots__ = ["key", "value"]
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    class BytesKwargsEntry(_message.Message):
        __slots__ = ["key", "value"]
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: bytes
        def __init__(self, key: _Optional[str] = ..., value: _Optional[bytes] = ...) -> None: ...
    MESSAGE_TYPE_FIELD_NUMBER: _ClassVar[int]
    REQUIRE_ANSWER_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    TASK_ID_FIELD_NUMBER: _ClassVar[int]
    PARENT_ID_FIELD_NUMBER: _ClassVar[int]
    FROM_UID_FIELD_NUMBER: _ClassVar[int]
    METHOD_NAME_FIELD_NUMBER: _ClassVar[int]
    NUMERIC_KWARGS_FIELD_NUMBER: _ClassVar[int]
    STR_KWARGS_FIELD_NUMBER: _ClassVar[int]
    BYTES_KWARGS_FIELD_NUMBER: _ClassVar[int]
    message_type: str
    require_answer: bool
    status: str
    task_id: str
    parent_id: str
    from_uid: str
    method_name: str
    numeric_kwargs: _containers.ScalarMap[str, float]
    str_kwargs: _containers.ScalarMap[str, str]
    bytes_kwargs: _containers.ScalarMap[str, bytes]
    def __init__(self, message_type: _Optional[str] = ..., require_answer: bool = ..., status: _Optional[str] = ..., task_id: _Optional[str] = ..., parent_id: _Optional[str] = ..., from_uid: _Optional[str] = ..., method_name: _Optional[str] = ..., numeric_kwargs: _Optional[_Mapping[str, float]] = ..., str_kwargs: _Optional[_Mapping[str, str]] = ..., bytes_kwargs: _Optional[_Mapping[str, bytes]] = ...) -> None: ...
