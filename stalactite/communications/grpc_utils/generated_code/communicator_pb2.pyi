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
    __slots__ = ("sender_id", "task_id", "method_name", "tensor_kwargs", "other_kwargs", "prometheus_metrics", "get_response_timeout")
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
    class PrometheusMetricsEntry(_message.Message):
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
    PROMETHEUS_METRICS_FIELD_NUMBER: _ClassVar[int]
    GET_RESPONSE_TIMEOUT_FIELD_NUMBER: _ClassVar[int]
    sender_id: str
    task_id: str
    method_name: str
    tensor_kwargs: _containers.ScalarMap[str, bytes]
    other_kwargs: _containers.ScalarMap[str, bytes]
    prometheus_metrics: _containers.ScalarMap[str, bytes]
    get_response_timeout: float
    def __init__(self, sender_id: _Optional[str] = ..., task_id: _Optional[str] = ..., method_name: _Optional[str] = ..., tensor_kwargs: _Optional[_Mapping[str, bytes]] = ..., other_kwargs: _Optional[_Mapping[str, bytes]] = ..., prometheus_metrics: _Optional[_Mapping[str, bytes]] = ..., get_response_timeout: _Optional[float] = ...) -> None: ...
