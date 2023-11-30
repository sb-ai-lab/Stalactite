import enum


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
