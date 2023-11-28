import enum


class ClientTask(enum.Enum):
    batched_exchange = 0
    exchange = 1


class PingResponse(str, enum.Enum):
    waiting_for_other_connections = 'waiting for other connections'
    all_ready = 'all ready'


class ClientStatus(enum.Enum):
    waiting = 0
    active = 1
    finished = 2


def format_important_logging(text: str) -> str:
    return "=" * 65 + "\n" + text + "\n" + "=" * 109
