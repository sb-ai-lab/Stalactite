import enum

from prometheus_client import Summary, Histogram

safetensor_exchange_time = Summary(
    'ExchangeBinarizedDataUnaryUnary_sec',
    'Time spent for full unary safetensor data exchange'
)

prototensor_exchange_time = Summary(
    'ExchangeNumpyDataUnaryUnary_sec',
    'Time spent for full unary proto data exchange'
)

safetensor_batch_exchange_time = Summary(
    'ExchangeBinarizedDataStreamStream_sec',
    'Time spent for batched bidirectional safetensor data exchange'
)

prototensor_batch_exchange_time = Summary(
    'ExchangeNumpyDataStreamStream_sec',
    'Time spent for batched bidirectional proto data exchange'
)

BUCKETS = (0.0001, 0.005, 0.01, 0.015, 0.02, 0.025, 0.05, 0.1, 0.2)
serialization_safetensor_time = Histogram(
    'save_data_safetensor_sec',
    'Time spent to form a request (safetensor)',
    buckets=BUCKETS,
)
deserialization_safetensor_time = Histogram(
    'load_data_safetensor_sec',
    'Time spent to load and unpack a response (safetensor)',
    buckets=BUCKETS,
)

serialization_proto_time = Histogram('save_data_proto_sec', 'Time spent to form a request (protobuf)', buckets=BUCKETS)
deserialization_proto_time = Histogram(
    'load_data_proto_sec',
    'Time spent to load and unpack a response (protobuf)',
    buckets=BUCKETS,
)


class Serialization(enum.Enum):
    safetensors = 0
    protobuf = 1


class ClientTask(enum.Enum):
    batched_exchange_tensor = 0
    exchange_tensor = 1
    batched_exchange_array = 2
    exchange_array = 3
    finish = 4


class PingResponse(str, enum.Enum):
    waiting_for_other_connections = 'waiting for other connections'
    all_ready = 'all ready'


class ClientStatus(enum.Enum):
    waiting = 0
    active = 1
    finished = 2


def format_important_logging(text: str) -> str:
    return "=" * 65 + "\n" + text + "\n" + "=" * 109
