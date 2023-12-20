import asyncio
import concurrent
from collections import defaultdict
import enum
import time
import uuid
from abc import ABC, abstractmethod
from queue import Queue, Empty
from dataclasses import dataclass
import logging
from typing import Optional, Dict, Union, AsyncIterator, List, Any, Iterator, Coroutine, cast
import threading
from concurrent import futures

import grpc
import torch
import safetensors.torch

from stalactite.base import (
    PartyMaster,
    PartyMember,
    PartyCommunicator,
    ParticipantFuture,
    Party,
    PartyDataTensor,
    RecordsBatch,
)
from stalactite.communications.helpers import ParticipantType, _Method
from stalactite.communications.grpc_utils.generated_code import communicator_pb2, communicator_pb2_grpc
from stalactite.communications.grpc_utils.utils import Status, ClientStatus, EventTask
from stalactite.communications.grpc_utils.grpc_servicer import GRpcCommunicatorServicer

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
sh = logging.StreamHandler()
sh.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(sh)


class MessageTypes(str, enum.Enum):
    server_task = 'server'
    client_task = 'client'


@dataclass
class ParticipantTasks:
    context: grpc.aio.ServicerContext
    queue: Queue


@dataclass
class MessageMaps:
    str_kwargs: dict[str, str] | None
    numeric_kwargs: dict[str, int | float] | None
    bytes_kwargs: dict[str, bytes] | None


def load_data(serialized_tensor: bytes) -> torch.Tensor:
    return safetensors.torch.load(serialized_tensor)['tensor']


def save_data(tensor: torch.Tensor):
    return safetensors.torch.save(tensors={'tensor': tensor})


def split_kwargs(kwargs: dict) -> MessageMaps:
    numeric_kwargs, str_kwargs, other_kwargs = dict(), dict(), dict()
    for key, value in kwargs.items():
        if isinstance(value, str):
            str_kwargs[key] = value
        elif isinstance(value, (int, float)):
            numeric_kwargs[key] = value
        else:
            if isinstance(value, torch.Tensor):
                other_kwargs[key] = save_data(value)
            else:
                raise ValueError(f'Unknown kwarg field type. Keyword arg name: {key}, type: {type(value)}')
    return MessageMaps(str_kwargs=str_kwargs, numeric_kwargs=numeric_kwargs, bytes_kwargs=other_kwargs)


def gather_kwargs(kwargs: MessageMaps) -> dict:
    other_kwargs = dict()
    for k, v in kwargs.bytes_kwargs.items():
        other_kwargs[k] = load_data(v)
    return {
        **kwargs.numeric_kwargs,
        **kwargs.str_kwargs,
        **other_kwargs,
    }


class GRpcMasterPartyCommunicator(PartyCommunicator, ABC):
    MEMBER_DATA_FIELDNAME = '__member_data__'

    def __init__(
            self,
            participant: PartyMaster,
            world_size: int,
            port: int | str,
            host: str,
            server_thread_pool_size: int = 10,
            max_message_size: int = -1,
    ):
        self.participant = participant
        self.world_size = world_size
        self.port = port
        self.host = host
        self.server_thread_pool_size = server_thread_pool_size
        self.max_message_size = max_message_size

        self.master_id = participant.id

        self.server_thread = None
        self.asyncio_event_loop = None
        self.servicer: GRpcCommunicatorServicer | None = None

    @property
    def status(self) -> Status:
        if self.servicer is None:
            return Status.not_started
        return self.servicer.status

    @property
    def number_connected_clients(self) -> int:
        if self.servicer is None:
            return 0
        return len(self.servicer.connected_clients)

    @property
    def members(self) -> list[str]:
        return list(self.servicer.connected_clients)

    def randezvous(self) -> None:
        if self.servicer is None:
            raise ValueError('Started rendezvous before initializing gRPC server and servicer')
        while self.status != Status.all_ready:
            time.sleep(0.1)
        if self.number_connected_clients != self.world_size:
            raise RuntimeError('Rendezvous failed')

    @property
    def is_ready(self) -> bool:
        return self.server_thread.running()

    def _get_tasks_from_main_queue(self) -> EventTask | None:
        try:
            return self.servicer.main_tasks_queue.get_nowait()
        except asyncio.QueueEmpty:
            return None

    def _put_to_main_tasks_queue(self, event: EventTask) -> None:
        self.servicer.main_tasks_queue.put_nowait(event)

    def _put_to_server_tasks_queue(self, event: EventTask, send_to_id: str) -> None:
        self.servicer.tasks_queue[send_to_id].put_nowait(event)

    @property
    def _server_tasks_futures(self) -> dict[str, ParticipantFuture]:
        return self.servicer.tasks_futures

    def _add_server_task_future(self, task_id: str, future: ParticipantFuture) -> None:
        self.servicer.tasks_futures[task_id] = future

    def send(self, send_to_id: str, method_name: str, require_answer: bool = True, **kwargs) -> ParticipantFuture:
        task_id = kwargs.pop('task_id', str(uuid.uuid4()))  # TODO ?
        parent_id = kwargs.pop('parent_id', None)  # TODO ?
        message_maps = split_kwargs(kwargs)

        future = ParticipantFuture(participant_id=send_to_id)

        task = EventTask(
            from_id=self.master_id,
            send_to_id=send_to_id,
            task_id=task_id,
            parent_id=parent_id,
            message=communicator_pb2.MainMessage(
                message_type=MessageTypes.server_task,
                status=self.status,
                require_answer=require_answer,
                task_id=task_id,
                parent_id=parent_id,
                from_uid=self.master_id,
                method_name=method_name,
                str_kwargs=message_maps.str_kwargs,
                numeric_kwargs=message_maps.numeric_kwargs,
                bytes_kwargs=message_maps.bytes_kwargs,
            )
        )
        if self.master_id == send_to_id:
            self._put_to_main_tasks_queue(task)
        else:
            self._put_to_server_tasks_queue(task, send_to_id=send_to_id)

        # not all command requires feedback
        if require_answer:
            self._add_server_task_future(task_id, future)
        else:
            future.set_result(None)

        logger.debug("Party communicator %s: sent to %s event %s" % (self.participant.id, send_to_id, task_id))
        return future

    def broadcast(self,
                  method_name: str,
                  mass_kwargs: Optional[List[Any]] = None,
                  participating_members: Optional[List[str]] = None,
                  parent_id: Optional[str] = None,
                  require_answer: bool = True,
                  include_current_participant: bool = False,
                  **kwargs) -> List[ParticipantFuture]:
        logger.debug("Sending event (%s) to all members" % method_name)
        members = participating_members or self.members
        unknown_members = set(members).difference(self.members)
        if len(unknown_members) > 0:
            raise ValueError(f"Unknown members: {unknown_members}. Existing members: {self.members}")

        for mkwargs in mass_kwargs:
            if not isinstance(mkwargs, torch.Tensor):
                raise ValueError(f"Only `list[torch.Tensor]` can be sent as `mass_kwargs` in broadcast operation.")

        if not mass_kwargs:
            mass_kwargs = [dict() for _ in members]
        elif mass_kwargs and len(mass_kwargs) != len(members):
            raise ValueError(f"Length of arguments list ({len(mass_kwargs)}) is not equal "
                             f"to the length of members ({len(members)})")
        else:
            mass_kwargs = [{self.MEMBER_DATA_FIELDNAME: args} for args in mass_kwargs]
        bc_futures = []
        for mkwargs, member_id in zip(mass_kwargs, members):
            if member_id == self.participant.id and not include_current_participant:
                continue
            future = self.send(
                send_to_id=member_id,
                method_name=method_name,
                require_answer=require_answer,
                parent_id=parent_id,
                **{**mkwargs, **kwargs}
            )
            bc_futures.append(future)
        return bc_futures

    def _run_coroutine(self, coroutine: Coroutine):
        self.asyncio_event_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.asyncio_event_loop)
        self.asyncio_event_loop.run_until_complete(coroutine)

    def run(self):
        try:
            self.servicer = GRpcCommunicatorServicer(
                world_size=self.world_size,
                master_id=self.master_id,
                host=self.host,
                port=self.port,
                threadpool_max_workers=self.server_thread_pool_size,
                max_message_size=self.max_message_size,
            )
            self.server_thread = threading.Thread(
                target=self._run_coroutine,
                args=(self.servicer.start_servicer_and_server(),),
                daemon=True
            )
            self.server_thread.start()

            self.randezvous()
            party = GRpcParty(party_communicator=self)

            event_loop = threading.Thread(target=self._run, daemon=True)
            event_loop.start()
            time.sleep(2)
            self.participant.run(party) # TODO
            # self.send(send_to_id=self.members[0], method_name=_Method.finalize, require_answer=False)
            self.send(send_to_id=self.master_id, method_name=_Method.finalize, require_answer=False)
            time.sleep(30)
            event_loop.join()
            self.server_thread.join(timeout=2.)
            logger.info("Party communicator %s: finished" % self.participant.id)
        except:
            logger.error("Exception in party communicator %s" % self.participant.id, exc_info=True)
            raise

    def _run(self):
        try:
            logger.info("Party communicator %s: starting event loop" % self.participant.id)

            while True:
                task = self._get_tasks_from_main_queue()
                if task is not None:
                    logger.debug("Party communicator %s: received event %s" % (self.participant.id, task.task_id))

                    event = task.message
                    event_data = gather_kwargs(
                        MessageMaps(
                            str_kwargs=dict(event.str_kwargs),
                            numeric_kwargs=dict(event.numeric_kwargs),
                            bytes_kwargs=dict(event.bytes_kwargs)
                        )
                    )

                    if event.method_name == _Method.service_return_answer.value:
                        if event.parent_id not in self._server_tasks_futures:
                            # todo: replace with custom error
                            raise ValueError(f"No awaiting future with id {event.parent_id}."
                                             f"(Participant id {self.participant.id}. "
                                             f"Event {event.task_id} from {event.from_uid})")

                        logger.debug(
                            "Party communicator %s: marking future %s as finished by answer of event %s"
                            % (self.participant.id, event.parent_id, event.task_id)
                        )

                        if 'result' not in event_data:
                            # todo: replace with custom error
                            raise ValueError("No result in data")

                        future = self._server_tasks_futures.pop(event.parent_id)
                        future.set_result(event_data['result'])
                        future.done()
                    elif event.method_name == _Method.service_heartbeat.value:
                        # TODO heartbeats are not tasks?
                        logger.info(
                            "Party communicator %s: received heartbeat from %s: %s"
                            % (self.participant.id, event.task_id, event_data)
                        )
                    elif event.method_name == _Method.finalize.value:
                        logger.info("Party communicator %s: finalized" % self.participant.id)
                        break
                    else:
                        raise ValueError(
                            f"Unsupported method {event.method_name} (Event {event.task_id} from {event.from_uid})"
                        )

            logger.info("Party communicator %s: finished event loop" % self.participant.id)
        except:
            logger.error("Exception in party communicator %s" % self.participant.id, exc_info=True)
            raise


class GRpcParty(Party):
    world_size: int
    members: List[str]

    def __init__(self, party_communicator: GRpcMasterPartyCommunicator, op_timeout: Optional[float] = None):
        self.party_communicator = party_communicator
        self.op_timeout = op_timeout

    @property
    def world_size(self) -> int:
        return self.party_communicator.world_size

    @property
    def members(self) -> List[str]:
        return self.party_communicator.members

    def _sync_broadcast_to_members(self, *,
                                   method_name: _Method,
                                   mass_kwargs: Optional[List[Any]] = None,
                                   participating_members: Optional[List[str]] = None,
                                   **kwargs) -> List[Any]:
        futures = self.party_communicator.broadcast(
            method_name=method_name,
            participating_members=participating_members,
            mass_kwargs=mass_kwargs,
            **kwargs
        )

        logger.debug("Party broadcast: waiting for answer to event %s (waiting for %s secs)"
                     % (method_name, self.op_timeout or "inf"))

        futures = concurrent.futures.wait(futures, timeout=self.op_timeout)
        completed_futures, uncompleted_futures \
            = cast(set[ParticipantFuture], futures[0]), cast(set[ParticipantFuture], futures[1])

        if len(uncompleted_futures) > 0:
            # todo: custom exception with additional info about uncompleted tasks
            raise RuntimeError(f"Not all tasks have been completed. "
                               f"Completed tasks: {len(completed_futures)}. "
                               f"Uncompleted tasks: {len(uncompleted_futures)}.")

        logger.debug("Party broadcast for event %s has succesfully finished" % method_name)

        fresults = {future.participant_id: future.result() for future in completed_futures}
        return [fresults[member_id] for member_id in self.party_communicator.members]

    def records_uids(self) -> List[List[str]]:
        return cast(List[List[str]], self._sync_broadcast_to_members(method_name=_Method.records_uids))

    def register_records_uids(self, uids: List[str]):
        self._sync_broadcast_to_members(method_name=_Method.register_records_uids, uids=uids)

    def initialize(self):
        self._sync_broadcast_to_members(method_name=_Method.initialize)

    def finalize(self):
        self._sync_broadcast_to_members(method_name=_Method.finalize)

    def update_weights(self, upd: PartyDataTensor):
        self._sync_broadcast_to_members(method_name=_Method.update_weights, mass_kwargs=upd)

    def predict(self, uids: List[str], use_test: bool = False) -> PartyDataTensor:
        return cast(
            PartyDataTensor,
            self._sync_broadcast_to_members(method_name=_Method.predict, uids=uids, use_test=True)
        )

    def update_predict(
            self,
            participating_members: List[str],
            batch: RecordsBatch,
            previous_batch: RecordsBatch,
            upd: PartyDataTensor
    ) -> PartyDataTensor:
        return cast(
            PartyDataTensor,
            self._sync_broadcast_to_members(
                method_name=_Method.update_predict,
                mass_kwargs=upd,
                participating_members=participating_members,
                batch=batch,
                previous_batch=previous_batch
            )
        )



class GRpcMemberPartyCommunicator(PartyCommunicator, ABC):
    MEMBER_DATA_FIELDNAME = '__member_data__'

    def __init__(
            self,
            participant: PartyMember,
            master_host: str,
            master_port: str,
            max_message_size: int = -1,
            heartbeat_interval: float = 2.,
            task_requesting_pings_interval: float = 3.,
            sent_task_timout: float = 3600.,
    ):
        self.participant = participant
        self.master_host = master_host
        self.master_port = master_port
        self.max_message_size = max_message_size
        self.heartbeat_interval = heartbeat_interval
        self.task_requesting_pings_interval = task_requesting_pings_interval
        self.sent_task_timout = sent_task_timout

        self._master_id = None
        self._world_status = None
        self._heartbeats_thread = None

        self.client_id = participant.id

        self.server_tasks_queue: Queue[communicator_pb2.MainMessage] = Queue()
        self.tasks_futures: dict[str, ParticipantFuture] = dict()

    def _start_client(self):
        self._grpc_channel = grpc.insecure_channel(
            f"{self.master_host}:{self.master_port}",
            options=[
                ('grpc.max_send_message_length', self.max_message_size),
                ('grpc.max_receive_message_length', self.max_message_size),
            ]
        )
        self._stub = communicator_pb2_grpc.CommunicatorStub(self._grpc_channel)
        logger.info("Starting ping-pong with the server")
        pingpong_responses = self._stub.Heartbeat(self._heartbeats(), wait_for_ready=True)
        self._heartbeats_thread = threading.Thread(
            name='heartbeat-thread',
            target=self._read_server_heartbeats,
            args=(pingpong_responses,),
            daemon=True,
        )
        self._heartbeats_thread.start()

    def _heartbeats(self):
        while True:
            time.sleep(self.heartbeat_interval)
            yield communicator_pb2.HB(agent_name=self.client_id, status=ClientStatus.alive)

    def _task_requests(self):
        while True:
            time.sleep(self.task_requesting_pings_interval)
            yield communicator_pb2.MainMessage(
                message_type=MessageTypes.client_task,
                from_uid=self.client_id,
            )

    def _read_server_heartbeats(self, server_responses: Iterator[communicator_pb2.HB]):
        for response in server_responses:
            logger.debug(f"Got pong from master: {response.agent_name}")
            self._world_status = response.status
            if self._master_id is None:
                self._master_id = response.agent_name
            else:
                assert self._master_id == response.agent_name, \
                    "Unexpected behaviour: Master id changed during the experiment"

    def _read_server_tasks(self, server_responses: Iterator[communicator_pb2.MainMessage]):
        while True:
            response = next(server_responses)
            logger.debug(f'Got task {response.task_id} from {response.from_uid}')
            self.server_tasks_queue.put(response)

    def randezvous(self) -> None:
        while self._world_status == Status.waiting:
            time.sleep(0.1)
        logger.info(f'Client {self.client_id} is ready to run')

    @property
    def is_ready(self) -> bool:
        return (self._master_id is not None) and (self._world_status == Status.all_ready)

    def _add_task_future(self, task_id: str, future: ParticipantFuture) -> None:
        self.tasks_futures[task_id] = future

    def send(self, send_to_id: str, method_name: str, require_answer: bool = True, **kwargs) -> ParticipantFuture:
        if not self.is_ready:
            raise RuntimeError("Cannot proceed because communicator is not ready. "
                               "Perhaps, rendezvous has not been called or was unsuccessful")

        if send_to_id not in (self._master_id, self.client_id):
            raise RuntimeError('GRpcMemberPartyCommunicator cannot send to other Members')

        task_id = kwargs.pop('task_id', str(uuid.uuid4())) # TODO remove duplicated code
        parent_id = kwargs.pop('parent_id', None)
        message_maps = split_kwargs(kwargs)

        future = ParticipantFuture(participant_id=send_to_id)

        task = EventTask(
            from_id=self.client_id,
            send_to_id=send_to_id,
            task_id=task_id,
            parent_id=parent_id,
            message=communicator_pb2.MainMessage(
                message_type=MessageTypes.client_task,
                status=ClientStatus.alive,
                require_answer=require_answer,
                task_id=task_id,
                parent_id=parent_id,
                from_uid=self.client_id,
                method_name=method_name,
                str_kwargs=message_maps.str_kwargs,
                numeric_kwargs=message_maps.numeric_kwargs,
                bytes_kwargs=message_maps.bytes_kwargs,
            )
        )
        if send_to_id == self._master_id:
            self._stub.UnaryExchange(task.message, timeout=self.sent_task_timout)
        else:
            self.server_tasks_queue.put(task.message)

        # not all command requires feedback
        if require_answer:
            self._add_task_future(task_id, future)
        else:
            future.set_result(None)

        logger.debug("Party communicator %s: sent to %s event %s" % (self.participant.id, send_to_id, task_id))
        return future

    def broadcast(self,
                  method_name: str,
                  mass_kwargs: Optional[List[Any]] = None,
                  participating_members: Optional[List[str]] = None,
                  parent_id: Optional[str] = None,
                  require_answer: bool = True,
                  include_current_participant: bool = False,
                  **kwargs) -> List[ParticipantFuture]:
        raise AttributeError('GRpcMemberPartyCommunicator cannot broadcast to other Members')

    def run(self):
        self._start_client()
        self.randezvous()
        self._start_receiving_tasks()
        self._run()
        logger.info(f"Party communicator {self.client_id} finished")

    def _start_receiving_tasks(self):
        server_task_responses = self._stub.BidiExchange(self._task_requests(), wait_for_ready=True)
        logger.info("Starting pinging server for the tasks")
        self.tasks_thread = threading.Thread(
            name='tasks-receive-thread',
            target=self._read_server_tasks,
            args=(server_task_responses,),
            daemon=True,
        )
        self.tasks_thread.start()

    def _run(self):
        logger.info("Party communicator %s: starting event loop" % self.participant.id)
        while True:
            task = self.server_tasks_queue.get()
            logger.debug("Party communicator %s: received event %s" % (self.participant.id, task.task_id))
            method = getattr(self.participant, task.method_name, None)
            if method is None:
                raise ValueError(f"Unsupported method {task.method_name} (Event {task.task_id} from {task.from_uid})")
            kwargs = gather_kwargs(
                MessageMaps(
                    bytes_kwargs=dict(task.bytes_kwargs),
                    str_kwargs=dict(task.str_kwargs),
                    numeric_kwargs=dict(task.numeric_kwargs),
                )
            )

            # TODO mkwargs
            # if self.MEMBER_DATA_FIELDNAME in kwargs:
            #     mkwargs = [kwargs[self.MEMBER_DATA_FIELDNAME]]
            #     del kwargs[self.MEMBER_DATA_FIELDNAME]
            # else:
            mkwargs = []
            result = method(*mkwargs, **kwargs)  # TODO put CPU heavy tasks in another thread?

            self.send(send_to_id=task.from_uid, method_name=_Method.service_return_answer,
                      parent_id=task.task_id, require_answer=False, result=result)

            if task.method_name == _Method.finalize.value:
                break

        logger.info("Party communicator %s: finished event loop" % self.participant.id)
