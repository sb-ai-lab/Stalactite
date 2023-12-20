import enum
from dataclasses import dataclass, field

from stalactite.communications.grpc_utils.generated_code import communicator_pb2

class Status(str, enum.Enum):
    not_started = 'not started'
    all_ready = 'all ready'
    waiting = 'waiting for others'
    finished = 'finished'


class ClientStatus(str, enum.Enum):
    alive = 'alive'



@dataclass
class EventTask:
    from_id: str
    send_to_id: str
    task_id: str
    message: communicator_pb2.MainMessage
    parent_id: str | None = field(default=None)
