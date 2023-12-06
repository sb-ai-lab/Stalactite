import enum

from prometheus_client import Summary, Histogram

# ====== Server endpoint metrics ======
safetensor_exchange = Summary('ExchangeBinarizedDataUnaryUnary_sec', 'Unary safetensor data exchange time')
prototensor_exchange = Summary('ExchangeNumpyDataUnaryUnary_sec', 'Unary proto data exchange time')
safetensor_batch_exchange = Summary(
    'ExchangeBinarizedDataStreamStream_sec', 'Batched bidirectional safetensor exchange time'
)
prototensor_batch_exchange = Summary('ExchangeNumpyDataStreamStream_sec', 'Batched bidirectional proto exchange time')

# ====== Client-side metrics ======
# generate_data = Summary('torch_rand_tensor_generation_sec', 'Data generation time')
collect_results_unary = Summary(
    'exchange_sec', 'Coroutine awaiting exchange task time', ['time_type', 'serialization']
)
collect_results_stream = Summary(
    'batched_exchange_sec', 'Coroutine awaiting batched_exchange task time', ['time_type', 'serialization']
)
# prototensor_collect_results_unary = Summary(
#     'exchange_array_sec', 'Coroutine awaiting exchange_array task time', ['time_type']
# )
# prototensor_collect_results_stream = Summary(
#     'batched_exchange_array_sec', 'Coroutine awaiting batched_exchange_array task time', ['time_type']
# )



# ====== Loading function metrics ======

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





class Serialization(str, enum.Enum):
    safetensors = 'safetensors'
    protobuf = 'protobuf'


class ClientTask(str, enum.Enum):
    batched_exchange_tensor = 'batched_exchange_tensor'
    exchange_tensor = 'exchange_tensor'
    batched_exchange_array = 'batched_exchange_array'
    exchange_array = 'exchange_array'
    finish = 'finish'


class PingResponse(str, enum.Enum):
    waiting_for_other_connections = 'waiting for other connections'
    all_ready = 'all ready'


class ClientStatus(enum.Enum):
    waiting = 0
    active = 1
    finished = 2


def format_important_logging(text: str) -> str:
    return "=" * 65 + "\n" + text + "\n" + "=" * 109
