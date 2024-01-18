from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class SendTime(_message.Message):
    __slots__ = ("task_id", "method_name", "send_time")
    TASK_ID_FIELD_NUMBER: _ClassVar[int]
    METHOD_NAME_FIELD_NUMBER: _ClassVar[int]
    SEND_TIME_FIELD_NUMBER: _ClassVar[int]
    task_id: str
    method_name: str
    send_time: float
    def __init__(self, task_id: _Optional[str] = ..., method_name: _Optional[str] = ..., send_time: _Optional[float] = ...) -> None: ...

class HB(_message.Message):
    __slots__ = ("agent_name", "status", "send_timings")
    AGENT_NAME_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    SEND_TIMINGS_FIELD_NUMBER: _ClassVar[int]
    agent_name: str
    status: str
    send_timings: _containers.RepeatedCompositeFieldContainer[SendTime]
    def __init__(self, agent_name: _Optional[str] = ..., status: _Optional[str] = ..., send_timings: _Optional[_Iterable[_Union[SendTime, _Mapping]]] = ...) -> None: ...

class MainMessage(_message.Message):
    __slots__ = ("message_type", "require_answer", "status", "task_id", "parent_id", "from_uid", "method_name", "parent_method_name", "parent_task_execution_time", "tensor_kwargs", "other_kwargs")
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
    MESSAGE_TYPE_FIELD_NUMBER: _ClassVar[int]
    REQUIRE_ANSWER_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    TASK_ID_FIELD_NUMBER: _ClassVar[int]
    PARENT_ID_FIELD_NUMBER: _ClassVar[int]
    FROM_UID_FIELD_NUMBER: _ClassVar[int]
    METHOD_NAME_FIELD_NUMBER: _ClassVar[int]
    PARENT_METHOD_NAME_FIELD_NUMBER: _ClassVar[int]
    PARENT_TASK_EXECUTION_TIME_FIELD_NUMBER: _ClassVar[int]
    TENSOR_KWARGS_FIELD_NUMBER: _ClassVar[int]
    OTHER_KWARGS_FIELD_NUMBER: _ClassVar[int]
    message_type: str
    require_answer: bool
    status: str
    task_id: str
    parent_id: str
    from_uid: str
    method_name: str
    parent_method_name: str
    parent_task_execution_time: float
    tensor_kwargs: _containers.ScalarMap[str, bytes]
    other_kwargs: _containers.ScalarMap[str, bytes]
    def __init__(self, message_type: _Optional[str] = ..., require_answer: bool = ..., status: _Optional[str] = ..., task_id: _Optional[str] = ..., parent_id: _Optional[str] = ..., from_uid: _Optional[str] = ..., method_name: _Optional[str] = ..., parent_method_name: _Optional[str] = ..., parent_task_execution_time: _Optional[float] = ..., tensor_kwargs: _Optional[_Mapping[str, bytes]] = ..., other_kwargs: _Optional[_Mapping[str, bytes]] = ...) -> None: ...
