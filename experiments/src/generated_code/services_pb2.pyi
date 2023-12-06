from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class SafetensorDataProto(_message.Message):
    __slots__ = ["data", "iteration", "client_id", "batch", "total_batches"]
    DATA_FIELD_NUMBER: _ClassVar[int]
    ITERATION_FIELD_NUMBER: _ClassVar[int]
    CLIENT_ID_FIELD_NUMBER: _ClassVar[int]
    BATCH_FIELD_NUMBER: _ClassVar[int]
    TOTAL_BATCHES_FIELD_NUMBER: _ClassVar[int]
    data: bytes
    iteration: int
    client_id: str
    batch: int
    total_batches: int
    def __init__(self, data: _Optional[bytes] = ..., iteration: _Optional[int] = ..., client_id: _Optional[str] = ..., batch: _Optional[int] = ..., total_batches: _Optional[int] = ...) -> None: ...

class Ping(_message.Message):
    __slots__ = ["data", "additional_info"]
    DATA_FIELD_NUMBER: _ClassVar[int]
    ADDITIONAL_INFO_FIELD_NUMBER: _ClassVar[int]
    data: str
    additional_info: str
    def __init__(self, data: _Optional[str] = ..., additional_info: _Optional[str] = ...) -> None: ...

class TensorProto(_message.Message):
    __slots__ = ["dims", "data_type", "float_data", "int32_data", "byte_data", "string_data", "double_data", "int64_data", "status", "iteration", "client_id", "batch", "total_batches"]
    class DataType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = []
        UNDEFINED: _ClassVar[TensorProto.DataType]
        FLOAT: _ClassVar[TensorProto.DataType]
        INT32: _ClassVar[TensorProto.DataType]
        BYTE: _ClassVar[TensorProto.DataType]
        STRING: _ClassVar[TensorProto.DataType]
        BOOL: _ClassVar[TensorProto.DataType]
        UINT8: _ClassVar[TensorProto.DataType]
        INT8: _ClassVar[TensorProto.DataType]
        UINT16: _ClassVar[TensorProto.DataType]
        INT16: _ClassVar[TensorProto.DataType]
        INT64: _ClassVar[TensorProto.DataType]
        FLOAT16: _ClassVar[TensorProto.DataType]
        DOUBLE: _ClassVar[TensorProto.DataType]
    UNDEFINED: TensorProto.DataType
    FLOAT: TensorProto.DataType
    INT32: TensorProto.DataType
    BYTE: TensorProto.DataType
    STRING: TensorProto.DataType
    BOOL: TensorProto.DataType
    UINT8: TensorProto.DataType
    INT8: TensorProto.DataType
    UINT16: TensorProto.DataType
    INT16: TensorProto.DataType
    INT64: TensorProto.DataType
    FLOAT16: TensorProto.DataType
    DOUBLE: TensorProto.DataType
    DIMS_FIELD_NUMBER: _ClassVar[int]
    DATA_TYPE_FIELD_NUMBER: _ClassVar[int]
    FLOAT_DATA_FIELD_NUMBER: _ClassVar[int]
    INT32_DATA_FIELD_NUMBER: _ClassVar[int]
    BYTE_DATA_FIELD_NUMBER: _ClassVar[int]
    STRING_DATA_FIELD_NUMBER: _ClassVar[int]
    DOUBLE_DATA_FIELD_NUMBER: _ClassVar[int]
    INT64_DATA_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    ITERATION_FIELD_NUMBER: _ClassVar[int]
    CLIENT_ID_FIELD_NUMBER: _ClassVar[int]
    BATCH_FIELD_NUMBER: _ClassVar[int]
    TOTAL_BATCHES_FIELD_NUMBER: _ClassVar[int]
    dims: _containers.RepeatedScalarFieldContainer[int]
    data_type: TensorProto.DataType
    float_data: _containers.RepeatedScalarFieldContainer[float]
    int32_data: _containers.RepeatedScalarFieldContainer[int]
    byte_data: bytes
    string_data: _containers.RepeatedScalarFieldContainer[bytes]
    double_data: _containers.RepeatedScalarFieldContainer[float]
    int64_data: _containers.RepeatedScalarFieldContainer[int]
    status: int
    iteration: int
    client_id: str
    batch: int
    total_batches: int
    def __init__(self, dims: _Optional[_Iterable[int]] = ..., data_type: _Optional[_Union[TensorProto.DataType, str]] = ..., float_data: _Optional[_Iterable[float]] = ..., int32_data: _Optional[_Iterable[int]] = ..., byte_data: _Optional[bytes] = ..., string_data: _Optional[_Iterable[bytes]] = ..., double_data: _Optional[_Iterable[float]] = ..., int64_data: _Optional[_Iterable[int]] = ..., status: _Optional[int] = ..., iteration: _Optional[int] = ..., client_id: _Optional[str] = ..., batch: _Optional[int] = ..., total_batches: _Optional[int] = ...) -> None: ...
