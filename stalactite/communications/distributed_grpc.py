from abc import ABC
import asyncio
import concurrent
from concurrent import futures
from copy import copy
import logging
from queue import Queue
import threading
import time
from typing import Any, Union, Optional, Iterator, Coroutine, cast
import uuid

import grpc
import torch

from stalactite.base import (
    PartyMaster,
    PartyMember,
    PartyCommunicator,
    ParticipantFuture,
    Party,
    PartyDataTensor,
    RecordsBatch,
)
from stalactite.communications.helpers import _Method, METHOD_VALUES
from stalactite.communications.grpc_utils.generated_code import communicator_pb2, communicator_pb2_grpc
from stalactite.communications.grpc_utils.grpc_servicer import GRpcCommunicatorServicer
from stalactite.communications.grpc_utils.utils import (
    Status,
    ClientStatus,
    MessageTypes,
    MethodMessage,
    PreparedTask,
    SerializedMethodMessage,
    prepare_kwargs,
    collect_kwargs,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
sh = logging.StreamHandler()
sh.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(sh)


class GRpcPartyCommunicator(PartyCommunicator, ABC):
    """ Base gRPC party communicator class. """
    participant: Union[PartyMaster, PartyMember]
    logging_level = logging.INFO

    def __init__(self, logging_level: Any = logging.INFO):
        logger.setLevel(logging_level)

    def prepare_task(
            self,
            message_type: MessageTypes,
            send_to_id: str,
            agent_status: str,
            method_name: _Method,
            task_id: Optional[str] = None,
            parent_id: Optional[str] = None,
            require_answer: bool = True,
            message: Optional[MethodMessage] = None,
    ) -> PreparedTask:
        """
        Form a message containing Task information.

        :param message_type: Type of the message to be sent
        :param send_to_id: Receiver identifier
        :param agent_status: Status of the sender
        :param method_name: Method name to execute on participant
        :param task_id: Unique identifier of the task
        :param parent_id: Unique identifier of the parent task
        :param require_answer: True if the task requires answer to sent back to sender
        :param message: Task data arguments
        """
        task_id = str(uuid.uuid4()) if task_id is None else task_id
        message_kwargs = prepare_kwargs(message)
        future = ParticipantFuture(participant_id=send_to_id)

        task_message = communicator_pb2.MainMessage(
            message_type=message_type,
            status=agent_status,
            require_answer=require_answer,
            task_id=task_id,
            parent_id=parent_id,
            from_uid=self.participant.id,
            method_name=method_name,
            tensor_kwargs=message_kwargs.tensor_kwargs,
            other_kwargs=message_kwargs.other_kwargs,
        )
        return PreparedTask(task_id=task_id, task_message=task_message, task_future=future)


class GRpcMasterPartyCommunicator(GRpcPartyCommunicator):
    """ gRPC Master communicator class. """
    MEMBER_DATA_FIELDNAME = '__member_data__'

    def __init__(
            self,
            participant: PartyMaster,
            world_size: int,
            port: Union[int, str],
            host: str,
            server_thread_pool_size: int = 10,
            max_message_size: int = -1,
            rendezvous_timeout: float = 3600.,
            **kwargs,
    ):
        """
        Initialize master communicator with connection parameters.

        :param participant: PartyMaster instance
        :param world_size: Number of VFL member agents
        :param port: Port of the gRPC server
        :param host: Host of the gRPC server
        :param server_thread_pool_size: Number of threadpool workers processing connections on the gRPC server
        :param max_message_size: Maximum message length that the gRPC channel can send or receive. -1 means unlimited
        :param rendezvous_timeout: Maximum time to wait until all the members are connected
        """
        self.participant = participant
        self.world_size = world_size
        self.port = port
        self.host = host
        self.server_thread_pool_size = server_thread_pool_size
        self.max_message_size = max_message_size
        self.rendezvous_timeout = rendezvous_timeout

        self.server_thread = None
        self.asyncio_event_loop = None
        self.servicer: Optional[GRpcCommunicatorServicer] = None

        super(GRpcMasterPartyCommunicator, self).__init__(**kwargs)

    @property
    def status(self) -> Status:
        """ Return status of the master communicator. """
        if self.servicer is None:
            return Status.not_started
        return self.servicer.status

    @property
    def number_connected_clients(self) -> int:
        """ Return number of VFL agent members connected to the server. """
        if self.servicer is None:
            return 0
        return len(self.servicer.connected_clients)

    @property
    def members(self) -> list[str]:
        """ List the VFL agent members` ids connected to the server. """
        return list(self.servicer.connected_clients)

    def randezvous(self) -> None:
        """ Wait until all the VFL agent members are connected to the gRPC server. """
        timer = time.time()
        if self.servicer is None:
            raise ValueError('Started rendezvous before initializing gRPC server and servicer')
        while self.status != Status.all_ready:
            time.sleep(0.1)
            if time.time() - timer > self.rendezvous_timeout:
                raise TimeoutError('Rendezvous timeout. You can try to set larger value in `rendezvous_timeout` param')
        if self.number_connected_clients != self.world_size:
            raise RuntimeError('Rendezvous failed')

    @property
    def is_ready(self) -> bool:
        """ Return True if the gRPC server is ready to accept connections and process the requests. """
        return self.server_thread.running()

    def _get_tasks_from_main_queue(self) -> Optional[communicator_pb2.MainMessage]:
        """ Get task scheduled for VFL master to process. """
        try:
            return self.servicer.main_tasks_queue.get_nowait()
        except asyncio.QueueEmpty:
            return None

    def _put_to_main_tasks_queue(self, task_message: communicator_pb2.MainMessage) -> None:
        """
        Schedule tasks for VFL master to perform.

        :param task_message: Message containing task information
        """
        self.servicer.main_tasks_queue.put_nowait(task_message)

    def _put_to_server_tasks_queue(self, task_message: communicator_pb2.MainMessage, send_to_id: str) -> None:
        """
        Schedule task to send from VFL master to VFL member.

        :param task_message: Message containing task information
        :param send_to_id: Identifier of the VFL member receiver
        """
        self.servicer.tasks_queue[send_to_id].put_nowait(task_message)

    @property
    def _server_tasks_futures(self) -> dict[str, ParticipantFuture]:
        """ Get futures of the launched tasks. """
        return self.servicer.tasks_futures

    def _add_server_task_future(self, task_id: str, future: ParticipantFuture) -> None:
        """
        Add future of the launched task.

        :param task_id: Unique identifier of the task
        :param future: Task future
        :return:
        """
        self.servicer.tasks_futures[task_id] = future

    def send(
            self,
            send_to_id: str,
            method_name: _Method,
            require_answer: bool = True,
            task_id: Optional[str] = None,
            parent_id: Optional[str] = None,
            message: Optional[MethodMessage] = None,
            **kwargs
    ) -> ParticipantFuture:
        """
        Send task to VFL member via gRPC channel.

        :param send_to_id: Identifier of the VFL member receiver
        :param method_name: Method name to execute on participant
        :param require_answer: True if the task requires answer to sent back to sender
        :param task_id: Unique identifier of the task
        :param parent_id: Unique identifier of the parent task
        :param message: Task data arguments
        :param kwargs: Optional kwargs which are ignored
        """
        if kwargs:
            logger.warning(f"Got unexpected kwargs in PartyCommunicator.sent method {kwargs}. Omitting.")

        prepared_task = self.prepare_task(
            message_type=MessageTypes.server_task,
            agent_status=self.status,
            send_to_id=send_to_id,
            method_name=method_name,
            require_answer=require_answer,
            task_id=task_id,
            parent_id=parent_id,
            message=message,
        )
        future = prepared_task.task_future
        task_id = prepared_task.task_id
        task_message = prepared_task.task_message

        if self.participant.id == send_to_id:
            self._put_to_main_tasks_queue(task_message)
        else:
            self._put_to_server_tasks_queue(task_message, send_to_id=send_to_id)

        # not all command requires feedback
        if require_answer:
            self._add_server_task_future(task_id, future)
        else:
            future.set_result(None)

        logger.debug("Party communicator %s: sent to %s event %s" % (self.participant.id, send_to_id, task_id))
        return future

    def broadcast(
            self,
            method_name: _Method,
            mass_kwargs: Optional[list[torch.Tensor]] = None,
            participating_members: Optional[list[str]] = None,
            parent_id: Optional[str] = None,
            require_answer: bool = True,
            include_current_participant: bool = False,
            message: Optional[MethodMessage] = None,
            **kwargs
    ) -> list[ParticipantFuture]:
        """
        Broadcast tasks to VFL agents via gRPC channel.

        :param method_name: Method name to execute on participant
        :param mass_kwargs: List of the torch.Tensors to send to each member in `participating_members`, respectively
        :param participating_members: List of the members to which the task will be broadcasted
        :param parent_id: Unique identifier of the parent task
        :param require_answer: True if the task requires answer to sent back to sender
        :param include_current_participant: True if the task should be performed by the VFL master too
        :param message: Task data arguments
        :param kwargs: Optional kwargs which are ignored
        """
        if kwargs:
            logger.warning(f"Got unexpected kwargs in PartyCommunicator.sent method {kwargs}. Omitting.")
        logger.debug("Sending event (%s) to all members" % method_name)
        members = participating_members or self.members
        unknown_members = set(members).difference(self.members)
        if len(unknown_members) > 0:
            raise ValueError(f"Unknown members: {unknown_members}. Existing members: {self.members}")

        if mass_kwargs is not None:
            for mkwargs in mass_kwargs:
                if not isinstance(mkwargs, torch.Tensor):
                    raise ValueError(f"Only `list[torch.Tensor]` can be sent as `mass_kwargs` in broadcast operation.")

        if mass_kwargs is None:
            mass_kwargs = [dict() for _ in members]
        elif mass_kwargs and len(mass_kwargs) != len(members):
            raise ValueError(
                f"Length of arguments list ({len(mass_kwargs)}) is not equal to the length of members ({len(members)})"
            )
        else:
            mass_kwargs = [{self.MEMBER_DATA_FIELDNAME: args} for args in mass_kwargs]

        bc_futures = []
        for mkwargs, member_id in zip(mass_kwargs, members):
            if member_id == self.participant.id and not include_current_participant:
                continue
            if message is not None:
                client_msg = copy(message)
                client_msg.tensor_kwargs.update(mkwargs)
            else:
                client_msg = MethodMessage()
            future = self.send(
                send_to_id=member_id,
                method_name=method_name,
                require_answer=require_answer,
                parent_id=parent_id,
                message=client_msg,
            )
            bc_futures.append(future)
        return bc_futures

    def _run_coroutine(self, coroutine: Coroutine):
        """ Run coroutine in the created asyncio event loop.
        Used to launch asyncio gRPC server in a separate thread.
        """
        self.asyncio_event_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.asyncio_event_loop)
        self.asyncio_event_loop.run_until_complete(coroutine)

    def run(self):
        """ Run the VFL master.
        Launch the gRPC server, wait until all the members are connected and start sending, receiving and processing
        tasks from the main loop.
        """
        try:
            self.servicer = GRpcCommunicatorServicer(
                world_size=self.world_size,
                master_id=self.participant.id,
                host=self.host,
                port=self.port,
                threadpool_max_workers=self.server_thread_pool_size,
                max_message_size=self.max_message_size,
                logging_level=self.logging_level,
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
            self.participant.run(party)
            self.send(send_to_id=self.participant.id, method_name=_Method.finalize, require_answer=False)
            event_loop.join()
            self.server_thread.join(timeout=2.)
            logger.info("Party communicator %s: finished" % self.participant.id)
        except:
            logger.error("Exception in party communicator %s" % self.participant.id, exc_info=True)
            raise

    def _run(self):
        """ Process tasks scheduled in the queue for VFL master. """
        try:
            logger.info("Party communicator %s: starting event loop" % self.participant.id)

            while True:
                task = self._get_tasks_from_main_queue()
                if task is not None:
                    logger.debug("Party communicator %s: received event %s" % (self.participant.id, task.task_id))

                    event_data = collect_kwargs(
                        SerializedMethodMessage(
                            tensor_kwargs=dict(task.tensor_kwargs),
                            other_kwargs=dict(task.other_kwargs),
                        )
                    )

                    if task.method_name == _Method.service_return_answer.value:
                        if task.parent_id not in self._server_tasks_futures:
                            # todo: replace with custom error
                            raise ValueError(f"No awaiting future with id {task.parent_id}."
                                             f"(Participant id {self.participant.id}. "
                                             f"Event {task.task_id} from {task.from_uid})")

                        logger.debug(
                            "Party communicator %s: marking future %s as finished by answer of event %s"
                            % (self.participant.id, task.parent_id, task.task_id)
                        )

                        if 'result' not in event_data:
                            # todo: replace with custom error
                            raise ValueError("No result in data")

                        future = self._server_tasks_futures.pop(task.parent_id)
                        future.set_result(event_data['result'])
                        future.done()
                    elif task.method_name == _Method.finalize.value:
                        logger.info("Party communicator %s: finalized" % self.participant.id)
                        break
                    else:
                        raise ValueError(
                            f"Unsupported method {task.method_name} (Event {task.task_id} from {task.from_uid})"
                        )

            logger.info("Party communicator %s: finished event loop" % self.participant.id)
        except:
            logger.error("Exception in party communicator %s" % self.participant.id, exc_info=True)
            raise


class GRpcParty(Party):
    """ Helper Party class for tasks scheduling. """
    world_size: int
    members: list[str]

    def __init__(self, party_communicator: GRpcMasterPartyCommunicator, op_timeout: Optional[float] = None):
        """
        Initialize GRpcParty with the communicator.

        :param party_communicator: VFL master communicator instance
        :param op_timeout: Maximum time to perform tasks in seconds
        """
        self.party_communicator = party_communicator
        self.op_timeout = op_timeout

    @property
    def world_size(self) -> int:
        """ Return number of the VFL members. """
        return self.party_communicator.world_size

    @property
    def members(self) -> list[str]:
        """ Return list of the VFL members identifiers. """
        return self.party_communicator.members

    def _sync_broadcast_to_members(
            self, *,
            method_name: _Method,
            mass_kwargs: Optional[list[Any]] = None,
            participating_members: Optional[list[str]] = None,
            parent_id: Optional[str] = None,
            require_answer: bool = True,
            include_current_participant: bool = False,
            message: Optional[MethodMessage] = None,
            **kwargs
    ) -> list[Any]:
        """
        Broadcast tasks to VFL agents using master communicator.

        :param method_name: Method name to execute on participant
        :param mass_kwargs: List of the torch.Tensors to send to each member in `participating_members`, respectively
        :param participating_members: List of the members to which the task will be broadcasted
        :param parent_id: Unique identifier of the parent task
        :param require_answer: True if the task requires answer to sent back to sender
        :param include_current_participant: True if the task should be performed by the VFL master too
        :param message: Task data arguments
        :param kwargs: Optional kwargs which are ignored
        """
        if kwargs:
            logger.warning(f"Got unexpected kwargs in PartyCommunicator.sent method {kwargs}. Omitting.")

        bc_futures = self.party_communicator.broadcast(
            method_name=method_name,
            participating_members=participating_members,
            mass_kwargs=mass_kwargs,
            parent_id=parent_id,
            require_answer=require_answer,
            include_current_participant=include_current_participant,
            message=message
        )

        logger.debug(
            "Party broadcast: waiting for answer to event %s (waiting for %s secs)"
            % (method_name, self.op_timeout or "inf")
        )

        bc_futures = concurrent.futures.wait(bc_futures, timeout=self.op_timeout)
        completed_futures, uncompleted_futures \
            = cast(set[ParticipantFuture], bc_futures[0]), cast(set[ParticipantFuture], bc_futures[1])

        if len(uncompleted_futures) > 0:
            # todo: custom exception with additional info about uncompleted tasks
            raise RuntimeError(f"Not all tasks have been completed. "
                               f"Completed tasks: {len(completed_futures)}. "
                               f"Uncompleted tasks: {len(uncompleted_futures)}.")

        logger.debug("Party broadcast for event %s has succesfully finished" % method_name)

        fresults = {future.participant_id: future.result() for future in completed_futures}
        return [fresults[member_id] for member_id in self.party_communicator.members]

    def records_uids(self) -> list[list[str]]:
        """ Collect records uids from VFL members. """
        return cast(list[list[str]], self._sync_broadcast_to_members(method_name=_Method.records_uids))

    def register_records_uids(self, uids: list[str]):
        """ Register records uids in VFL agents. """
        self._sync_broadcast_to_members(
            method_name=_Method.register_records_uids,
            message=MethodMessage(other_kwargs={'uids': uids})
        )

    def initialize(self):
        """ Initialize party communicators. """
        self._sync_broadcast_to_members(method_name=_Method.initialize)

    def finalize(self):
        """ Finilize party communicators. """
        self._sync_broadcast_to_members(method_name=_Method.finalize)

    def update_weights(self, upd: PartyDataTensor):
        """ Update model`s weights on agents. """
        self._sync_broadcast_to_members(
            method_name=_Method.update_weights,
            mass_kwargs=upd,
        )

    def predict(self, uids: list[str], use_test: bool = False) -> PartyDataTensor:
        """ Make predictions using VFL agents` models. """
        return cast(
            PartyDataTensor,
            self._sync_broadcast_to_members(
                method_name=_Method.predict,
                message=MethodMessage(other_kwargs={'uids': uids, 'use_test': True})
            )
        )

    def update_predict(
            self,
            participating_members: list[str],
            batch: RecordsBatch,
            previous_batch: RecordsBatch,
            upd: PartyDataTensor
    ) -> PartyDataTensor:
        """ Perform update_predict operation on the VFL agents. """
        return cast(
            PartyDataTensor,
            self._sync_broadcast_to_members(
                method_name=_Method.update_predict,
                mass_kwargs=upd,
                participating_members=participating_members,
                message=MethodMessage(other_kwargs={'batch': batch, 'previous_batch': previous_batch})
            )
        )


class GRpcMemberPartyCommunicator(GRpcPartyCommunicator):
    """ gRPC Master communicator class. """
    MEMBER_DATA_FIELDNAME = '__member_data__'

    def __init__(
            self,
            participant: PartyMember,
            master_host: str,
            master_port: str,
            max_message_size: int = -1,
            heartbeat_interval: float = 2.,
            task_requesting_pings_interval: float = 0.1,
            sent_task_timout: float = 3600.,
            rendezvous_timeout: float = 3600.,
            **kwargs,
    ):
        """
        Initialize member communicator with connection and task parameters.

        :param participant: PartyMember instance
        :param master_port: Port of the gRPC server
        :param master_host: Host of the gRPC server
        :param max_message_size: Maximum message length that the gRPC channel can send or receive. -1 means unlimited
        :param heartbeat_interval: Interval in seconds between heartbeats from member to master
        :param task_requesting_pings_interval: Interval in seconds between task requests from master
        :param sent_task_timout: Time to wait before task is marked `acknowledged` by the master
        :param rendezvous_timeout: Maximum time to wait until all the members are connected
        """
        self.participant = participant
        self.master_host = master_host
        self.master_port = master_port
        self.max_message_size = max_message_size
        self.heartbeat_interval = heartbeat_interval
        self.task_requesting_pings_interval = task_requesting_pings_interval
        self.sent_task_timout = sent_task_timout
        self.rendezvous_timeout = rendezvous_timeout

        self._master_id = None
        self._world_status = None
        self._heartbeats_thread = None

        self.server_tasks_queue: Queue[communicator_pb2.MainMessage] = Queue()
        self.tasks_futures: dict[str, ParticipantFuture] = dict()

        super(GRpcMemberPartyCommunicator, self).__init__(**kwargs)


    def _start_client(self):
        """ Create a gRPC channel. Start a thread with heartbeats from the member. """
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
        """ Generate heartbeats messages. """
        while True:
            time.sleep(self.heartbeat_interval)
            yield communicator_pb2.HB(agent_name=self.participant.id, status=ClientStatus.alive)

    def _task_requests(self):
        """ Generate tasks requests to the master. """
        while True:
            time.sleep(self.task_requesting_pings_interval)
            yield communicator_pb2.MainMessage(
                message_type=MessageTypes.client_task,
                from_uid=self.participant.id,
            )

    def _read_server_heartbeats(self, server_responses: Iterator[communicator_pb2.HB]):
        """ Read responses to heartbeats from master.
        Update info on the world status.

        :param server_responses: Iterator of the server responses
        """
        for response in server_responses:
            logger.debug(f"Got pong from master: {response.agent_name}")
            self._world_status = response.status
            if self._master_id is None:
                self._master_id = response.agent_name
            else:
                assert self._master_id == response.agent_name, \
                    "Unexpected behaviour: Master id changed during the experiment"

    def _read_server_tasks(self, server_responses: Iterator[communicator_pb2.MainMessage]):
        """ Read responses to task requests from master.
        Update tasks queue to process tasks in a main thread.

        :param server_responses: Iterator of the server responses
        """
        for response in server_responses:
            logger.debug(f'Got task {response.task_id} from {response.from_uid}')
            self.server_tasks_queue.put(response)

    def randezvous(self) -> None:
        """ Wait until VFL master identify readiness to start of all the VFL members. """
        timer = time.time()
        while self._world_status != Status.all_ready:
            time.sleep(0.1)
            if time.time() - timer > self.rendezvous_timeout:
                raise TimeoutError('Rendezvous timeout. You can try to set larger value in `rendezvous_timeout` param')
        logger.info(f'Client {self.participant.id} is ready to run')

    @property
    def is_ready(self) -> bool:
        """ Return True if the VFL master is found and other members are alive. """
        return (self._master_id is not None) and (self._world_status == Status.all_ready)

    def _add_task_future(self, task_id: str, future: ParticipantFuture) -> None:
        """
        Add future of the launched task.

        :param task_id: Unique identifier of the task
        :param future: Task future
        :return:
        """
        self.tasks_futures[task_id] = future

    def send(
            self,
            send_to_id: str,
            method_name: _Method,
            require_answer: bool = True,
            task_id: Optional[str] = None,
            parent_id: Optional[str] = None,
            message: Optional[MethodMessage] = None,
            **kwargs
    ) -> ParticipantFuture:
        """
        Send task to VFL master via gRPC channel.

        :param send_to_id: Identifier of the VFL receiver (only master is available for member`s send
        :param method_name: Method name to execute on participant
        :param require_answer: True if the task requires answer to sent back to sender
        :param task_id: Unique identifier of the task
        :param parent_id: Unique identifier of the parent task
        :param message: Task data arguments
        :param kwargs: Optional kwargs which are ignored
        """
        if kwargs:
            logger.warning(f"Got unexpected kwargs in PartyCommunicator.sent method {kwargs}. Omitting.")

        if not self.is_ready:
            raise RuntimeError("Cannot proceed because communicator is not ready. "
                               "Perhaps, rendezvous has not been called or was unsuccessful")

        if send_to_id not in (self._master_id, self.participant.id):
            raise RuntimeError('GRpcMemberPartyCommunicator cannot send to other Members')

        prepared_task = self.prepare_task(
            message_type=MessageTypes.client_task,
            agent_status=ClientStatus.alive,
            send_to_id=send_to_id,
            method_name=method_name,
            require_answer=require_answer,
            task_id=task_id,
            parent_id=parent_id,
            message=message,
        )
        future = prepared_task.task_future
        task_id = prepared_task.task_id
        task_message = prepared_task.task_message

        if send_to_id == self._master_id:
            res = self._stub.UnaryExchange(task_message, timeout=self.sent_task_timout)
            assert res.message_type == MessageTypes.acknowledgment, 'Sent message was not acknowledged'
        else:
            self.server_tasks_queue.put(task_message)

        # not all command requires feedback
        if require_answer:
            self._add_task_future(task_id, future)
        else:
            future.set_result(None)

        logger.debug("Party communicator %s: sent to %s event %s" % (self.participant.id, send_to_id, task_id))
        return future

    def broadcast(self,
                  method_name: _Method,
                  mass_kwargs: Optional[list[Any]] = None,
                  participating_members: Optional[list[str]] = None,
                  parent_id: Optional[str] = None,
                  require_answer: bool = True,
                  include_current_participant: bool = False,
                  **kwargs) -> list[ParticipantFuture]:
        """ Broadcast task to VFL agents via gRPC channel.
        This method is unavailable for GRpcMemberPartyCommunicator as it cannot communcate with other members.
        """
        raise AttributeError('GRpcMemberPartyCommunicator cannot broadcast to other Members')

    def run(self):
        """ Run the VFL member.
        Start the gRPC client threads, wait until server sends an `all ready` heartbeat response. Start requesting,
        receiving and processing tasks from the VFL master.
        """
        self._start_client()
        self.randezvous()
        self._start_receiving_tasks()
        self._run()
        logger.info(f"Party communicator {self.participant.id} finished")

    def _start_receiving_tasks(self):
        """ Start a thread with tasks requests to VFL master. """
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
        """ Process tasks scheduled in the queue for VFL member. """
        logger.info("Party communicator %s: starting event loop" % self.participant.id)
        while True:
            task = self.server_tasks_queue.get()
            logger.debug("Party communicator %s: received event %s" % (self.participant.id, task.task_id))

            method = getattr(self.participant, task.method_name, None)
            if method is None:
                raise ValueError(f"Unsupported method {task.method_name} (Event {task.task_id} from {task.from_uid})")

            kwargs = collect_kwargs(
                SerializedMethodMessage(
                    tensor_kwargs=dict(task.tensor_kwargs),
                    other_kwargs=dict(task.other_kwargs),
                )
            )
            mkwargs = kwargs.pop(self.MEMBER_DATA_FIELDNAME, None)
            if mkwargs is not None:
                if isinstance(mkwargs, torch.Tensor):
                    mkwargs = [mkwargs]
                else:
                    raise ValueError('Mass kwargs must be torch.Tensor')
            else:
                mkwargs = []

            result = method(*mkwargs, **kwargs)  # TODO put CPU heavy tasks in another thread?
            client_msg = MethodMessage()
            setattr(client_msg, METHOD_VALUES[task.method_name], {'result': result})

            self.send(
                send_to_id=task.from_uid,
                method_name=_Method.service_return_answer,
                parent_id=task.task_id,
                require_answer=False,
                message=client_msg
            )

            if task.method_name == _Method.finalize.value:
                break

        logger.info("Party communicator %s: finished event loop" % self.participant.id)
