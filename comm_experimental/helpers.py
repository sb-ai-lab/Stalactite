import enum


class PingResponse(str, enum.Enum):
    waiting_for_other_connections = 'waiting for other connections'
    all_ready = 'all ready'


class ClientStatus(enum.Enum):
    waiting = 0
    active = 1
    finished = 2
