import collections
import enum
import itertools
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, List, Optional, Union

import torch

from stalactite.base import (
    PartyMember,
    PartyMaster,
    PartyCommunicator,
    Method,
    MethodKwargs,
    Task,
    DataTensor,
    RecordsBatch,
    Batcher,
    PartyAgent,
)

logger = logging.getLogger(__name__)

logger.setLevel(logging.DEBUG)
sh = logging.StreamHandler()
sh.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
logger.addHandler(sh)


@dataclass
class Keys:
    public: Any
    private: Any = None


class Role(str, enum.Enum):
    arbiter = "arbiter"
    master = "master"
    member = "member"


class ArbiteredMethod(Method):
    get_public_key = "get_public_key"


class SecurityProtocol(ABC):
    """ Base proxy class for Homomorphic Encryption (HE) protocol. """
    _keys: Optional[Keys] = None

    @abstractmethod
    def encrypt(self, data: torch.Tensor) -> Any:
        """ Encrypt data using public key.

        :param data: Data to encrypt using the public key.
        :return: Encrypted data.
        """

        ...

    @abstractmethod
    def drop_private_key(self) -> Keys:
        """ Get private-public keys pair and return only the public key.

        :return: Private-public keys pair with private key as None.
        """
        ...

    @property
    def public_key(self) -> Keys:
        """ Public key getter.

        :return: Private-public keys pair with private key as None.
        """
        if self._keys is not None:
            keys = self.drop_private_key()
            if keys.private is not None:
                raise RuntimeError('Error while getting public key.')
            return keys
        else:
            raise RuntimeError(
                'No public-private key pair was initialized. You should call the `generate_keys` method on arbiter.'
            )

    @property
    def keys(self) -> Keys:
        """ Getter for the public (required) - private (optional) keys pair."""
        return self._keys

    @keys.setter
    def keys(self, keys: Keys) -> None:
        """ Setter for the public (required) - private (optional) keys pair."""

        if keys.public is None:
            raise ValueError('Trying to set keys which do not contain public key.')
        self._keys = keys


class SecurityProtocolArbiter(SecurityProtocol, ABC):
    """ Expanded functionality (trusted party arbiter) base proxy class for Homomorphic Encryption (HE) protocol. """

    @abstractmethod
    def generate_keys(self):
        """ Generate private and public keys. """
        ...

    @abstractmethod
    def decrypt(self, decrypted_data: Any) -> torch.Tensor:
        """ Decrypt data using private key.

        :param decrypted_data: Data to decrypt using the private key.
        :return: Decrypted data.
        """
        ...


class PartyArbiter(PartyAgent, ABC):
    security_protocol: SecurityProtocolArbiter
    id: str

    def is_ready(self, party: PartyCommunicator):
        if party.master is not None and party.members is not None:
            return True
        else:
            return False

    @abstractmethod
    @property
    def batcher(self) -> Batcher:
        ...

    def get_public_key(self) -> Keys:
        return self.security_protocol.public_key

    def run(self, party: PartyCommunicator) -> None:
        logger.info("Running arbiter %s" % self.id)
        initialize_task = party.recv(Task(method_name=Method.initialize, from_id=party.master, to_id=self.id))
        self.execute_received_task(initialize_task)

        register_records_uids_task = party.recv(
            Task(method_name=Method.register_records_uids, from_id=party.master, to_id=self.id)
        )
        self.execute_received_task(register_records_uids_task)

        public_keys_tasks = [
            Task(ArbiteredMethod.get_public_key, from_id=agent, to_id=self.id)
            for agent in party.members + [party.master]
        ]

        pk_tasks = party.gather(public_keys_tasks, recv_results=False)

        keys = [self.execute_received_task(task) for task in pk_tasks]
        party.scatter(
            method_name=ArbiteredMethod.get_public_key,
            method_kwargs=[
                MethodKwargs(other_kwargs={'result': key})
                for key in keys
            ]
        )

        self.loop(batcher=self.batcher, party=party)

        finalize_task = party.recv(Task(method_name=Method.finalize, from_id=party.master, to_id=self.id))
        self.execute_received_task(finalize_task)
        logger.info("Finished master %s" % self.id)


class ArbiteredPartyMaster(PartyMaster, PartyMember, ABC):
    security_protocol: Optional[SecurityProtocol] = None

    def run(self, party: PartyCommunicator) -> None:
        logger.info("Running master %s" % self.id)

        records_uids_tasks = party.broadcast(
            ArbiteredMethod.records_uids,
            participating_members=party.members,
        )

        records_uids_results = party.gather(records_uids_tasks, recv_results=True)

        collected_uids_results = [task.result for task in records_uids_results]

        party.broadcast(
            ArbiteredMethod.initialize,
            participating_members=party.members + [party.arbiter],
        )
        self.initialize()

        uids = self.synchronize_uids(collected_uids_results, world_size=party.world_size)
        party.broadcast(
            ArbiteredMethod.register_records_uids,
            method_kwargs=MethodKwargs(other_kwargs={"uids": uids}),
            participating_members=party.members + [party.master] + [party.arbiter],
        )

        register_records_uids_task = party.recv(
            Task(method_name=Method.register_records_uids, from_id=party.master, to_id=self.id)
        )
        self.execute_received_task(register_records_uids_task)

        pk_task = party.send(method_name=ArbiteredMethod.get_public_key,
                             send_to_id=party.arbiter)  # TODO move to execute_task
        pk = party.recv(pk_task, recv_results=True).result

        self.security_protocol.keys = pk.result

        self.loop(batcher=self.make_batcher(uids=uids, party_members=party.members), party=party)

        party.broadcast(
            Method.finalize,
            participating_members=party.members + [party.arbiter],
        )
        self.finalize()
        logger.info("Finished master %s" % self.id)


class ArbiteredPartyMember(PartyMember, ABC):
    security_protocol: Optional[SecurityProtocol] = None

    def run(self, party: PartyCommunicator) -> None:
        logger.info("Running member %s" % self.id)

        synchronize_uids_task = party.recv(
            Task(method_name=Method.records_uids, from_id=self.master_id, to_id=self.id)
        )
        uids = self.execute_received_task(synchronize_uids_task)
        party.send(self.master_id, Method.records_uids, result=uids)

        initialize_task = party.recv(Task(method_name=Method.initialize, from_id=self.master_id, to_id=self.id))
        self.execute_received_task(initialize_task)

        register_records_uids_task = party.recv(
            Task(method_name=Method.register_records_uids, from_id=self.master_id, to_id=self.id)
        )

        self.execute_received_task(register_records_uids_task)

        pk_task = party.send(method_name=ArbiteredMethod.get_public_key,
                             send_to_id=party.arbiter)  # TODO move to execute_task
        pk = party.recv(pk_task, recv_results=True).result

        self.security_protocol.keys = pk.result

        self.loop(batcher=self.batcher, party=party)

        finalize_task = party.recv(Task(method_name=Method.finalize, from_id=self.master_id, to_id=self.id))
        self.execute_received_task(finalize_task)
        logger.info("Finished member %s" % self.id)

    # @abstractmethod
    # def loop(self, batcher: Batcher, party: ArbiterCommunicator):
    #     for titer in batcher:
    #         iter_start_time = time.time()
    #         upd_task = party.gather(
    #             task=Task(ArbiterMethod.compute_updates),
    #             recv_results=False,
    #         )
    #         participants_gradients_enc = self.aggregate(upd_task)
    #         participants_gradients = self.security_protocol.decrypt(participants_gradients_enc)
    #         participant_updates = self.update_global_model(participants_gradients)
    #         party.scatter(ArbiterMethod.compute_updates, participant_updates)

# class PartyMasterArbiter(PartyMaster, ABC):
#     """ Abstract base class for the master party in the VFL experiment. """
#
#     id: str
#     epochs: int
#     report_train_metrics_iteration: int
#     report_test_metrics_iteration: int
#     target: DataTensor
#     target_uids: List[str]
#     test_target: DataTensor
#
#     security_protocol: SecurityProtocol
#
#     _public_key = None
#
#     _iter_time: list[tuple[int, float]] = list()
#
#     @property
#     def train_timings(self) -> list:
#         """ Return list of tuples representing iteration timings from the main loop. """
#         return self._iter_time
#
#
#     def save_pubkey(self, public_key: Keys):
#         self._public_key = public_key
#
#     def encrypt(self, data: torch.Tensor) -> Any:
#         return self.security_protocol.encrypt(self._public_key, data)
#
#     def loop(self, batcher: Batcher, party: PartyCommunicator) -> None:
#         """ Run main training loop on the VFL master.
#
#         :param batcher: Batcher for creating training batches.
#         :param communicator: Communicator instance used for communication between VFL agents.
#
#         :return: None
#         """
#         logger.info("Master %s: entering training loop" % self.id)
#
#         for titer in batcher:
#             iter_start_time = time.time()
#             participant_partial_pred_tasks = party.broadcast(Method.partial_prediction, participating_members=party.members + [party.master])
#             predict_task = party.recv(Task(Method.partial_prediction), recv_results=False)
#             master_partial_preds = self.partial_prediction(titer, predict_task)
#
#             party.send(send_to_id=party.master, method_name=Method.partial_prediction, method_kwargs=master_partial_preds)
#             participant_partial_predictions = party.gather(participant_partial_pred_tasks, recv_results=True)
#
#             loss = self.aggregate_partial_preds(participant_partial_predictions) # TODO error ?
#             party.broadcast(Method.compute_gradient, method_kwargs=loss, participating_members=party.members + [party.master])
#             calculate_grad_enc_task = party.recv(Task(Method.compute_gradient), recv_results=False)
#             gradient_enc = self.calculate_grad_enc(calculate_grad_enc_task)
#
#             upd_task = party.send(
#                 send_to_id=party.arbiter,
#                 task=Task(ArbiterMethod.compute_updates, method_kwargs=gradient_enc)
#             )
#             upd = party.recv(upd_task, recv_results=True)
#
#             self.update_weights(upd=upd)
#
#
#             # if self.report_train_metrics_iteration > 0 and titer.seq_num % self.report_train_metrics_iteration == 0:
#             #     predict_tasks = party.broadcast(Method.predict, participating_members=titer.participating_members + master)
#             #     task = party.recv(Method.predict)
#             #     predict = execute_task(task)
#             #     predict_enc = self.encrypt(predict)
#             #     party.send(send_to_id=master, predict_enc)
#             #     party_predictions = [task.result for task in party.gather(predict_tasks, recv_results=True)]
#
#             #     predictions_enc = self.aggregate(party.members, party_predictions, infer=True)
#             #     decode_preds = party.send(ArbiterMethod.decrypt_data, predictions, send_to_id=party.arbiter)
#             #     predictions = party.recv(decode_preds, recv_results=True)
#             #     self.report_metrics(self.target, predictions, name="Train")
#
#             # if self.report_test_metrics_iteration > 0 and titer.seq_num % self.report_test_metrics_iteration == 0:
#             #     predict_tasks = party.broadcast(Method.predict, use_Test=True, participating_members=titer.participating_members + master)
#             #     task = party.recv(Method.predict)
#             #     predict = execute_task(task)
#             #     predict_enc = self.encrypt(predict)
#             #     party.send(send_to_id=master, predict_enc)
#             #     party_predictions = [task.result for task in party.gather(predict_tasks, recv_results=True)]
#
#             #     predictions_enc = self.aggregate(party.members, party_predictions, infer=True)
#             #     decode_preds = party.send(ArbiterMethod.decrypt_data, predictions, send_to_id=party.arbiter)
#             #     predictions = party.recv(decode_preds, recv_results=True)
#             #     self.report_metrics(self.target, predictions, name="Test")
#             self._iter_time.append((titer.seq_num, time.time() - iter_start_time))
#
#
#
#
#     def synchronize_uids(self, collected_uids: list[list[str]], world_size: int) -> List[str]:
#         """ Synchronize unique records identifiers across party members.
#
#         :param collected_uids: List of lists containing unique records identifiers collected from party members.
#         :param world_size: Number of party members in the experiment.
#
#         :return: Common records identifiers among the agents used in training loop.
#         """
#         logger.debug("Master %s: synchronizing uids for party of size %s" % (self.id, world_size))
#         uids = itertools.chain(self.target_uids, (uid for member_uids in collected_uids for uid in set(member_uids)))
#         shared_uids = sorted([uid for uid, count in collections.Counter(uids).items() if count == world_size + 1])
#         logger.debug("Master %s: registering shared uids f size %s" % (self.id, len(shared_uids)))
#         set_shared_uids = set(shared_uids)
#         uid2idx = {uid: i for i, uid in enumerate(self.target_uids) if uid in set_shared_uids}
#         selected_tensor_idx = [uid2idx[uid] for uid in shared_uids]
#
#         self.target = self.target[selected_tensor_idx]
#         self.target_uids = shared_uids
#         logger.debug("Master %s: record uids has been successfully synchronized")
#         return shared_uids
#
#     @abstractmethod
#     def make_batcher(self, uids: List[str], party_members: List[str]) -> Batcher:
#         """ Make a batcher for training.
#
#         :param uids: List of unique identifiers of dataset records.
#         :param party_members: List of party members` identifiers.
#
#         :return: Batcher instance.
#         """
#         ...
#
#     @abstractmethod
#     def initialize(self):
#         """ Initialize the party master. """
#         ...
#
#     @abstractmethod
#     def finalize(self):
#         """ Finalize the party master. """
#         ...
#
#     @abstractmethod
#     def make_init_updates(self, world_size: int) -> PartyDataTensor:
#         """ Make initial updates for party members.
#
#         :param world_size: Number of party members.
#
#         :return: Initial updates as a list of tensors.
#         """
#         ...
#
#     @abstractmethod
#     def aggregate(
#             self, participating_members: List[str], party_predictions: PartyDataTensor, infer: bool = False
#     ) -> DataTensor:
#         """ Aggregate members` predictions.
#
#         :param participating_members: List of participating party member identifiers.
#         :param party_predictions: List of party predictions.
#         :param infer: Flag indicating whether to perform inference.
#
#         :return: Aggregated predictions.
#         """
#         ...
#
#     @abstractmethod
#     def compute_updates(
#             self,
#             participating_members: List[str],
#             predictions: DataTensor,
#             party_predictions: PartyDataTensor,
#             world_size: int,
#             subiter_seq_num: int,
#     ) -> List[DataTensor]:
#         """ Compute updates based on members` predictions.
#
#         :param participating_members: List of participating party member identifiers.
#         :param predictions: Model predictions.
#         :param party_predictions: List of party predictions.
#         :param world_size: Number of party members.
#         :param subiter_seq_num: Sub-iteration sequence number.
#
#         :return: List of updates as tensors.
#         """
#         ...
#
#     @abstractmethod
#     def report_metrics(self, y: DataTensor, predictions: DataTensor, name: str):
#         """ Report metrics based on target values and predictions.
#
#         :param y: Target values.
#         :param predictions: Model predictions.
#         :param name: Name of the dataset ("Train" or "Test").
#
#         :return: None
#         """
#         ...
#
#
# class PartyMemberArbiter(ABC):
#     """ Abstract base class for the member party in the VFL experiment. """
#
#     id: str
#     master_id: str
#     report_train_metrics_iteration: int
#     report_test_metrics_iteration: int
#     _iter_time: list[tuple[int, float]] = list()
#
#     def _execute_received_task(self, task: Task) -> Optional[Union[DataTensor, List[str]]]:
#         """ Execute received method on a member.
#
#         :param task: Received task to execute.
#         :return: Execution result.
#         """
#         return getattr(self, task.method_name)(**task.kwargs_dict)
#
#     def run(self, party: PartyCommunicator):
#         """ Run the VFL experiment with the member party.
#
#         Current method implements initialization of the member, launches the main training loop,
#         and finalizes the member.
#
#         :param communicator: Communicator instance used for communication between VFL agents.
#         :return: None
#         """
#         logger.info("Running member %s" % self.id)
#
#         synchronize_uids_task = party.recv(
#             Task(method_name=Method.records_uids, from_id=self.master_id, to_id=self.id)
#         )
#         uids = self._execute_received_task(synchronize_uids_task)
#         party.send(self.master_id, Method.records_uids, result=uids)
#         initialize_task = party.recv(Task(method_name=Method.initialize, from_id=self.master_id, to_id=self.id))
#         self._execute_received_task(initialize_task)
#         register_records_uids_task = party.recv(
#             Task(method_name=Method.register_records_uids, from_id=self.master_id, to_id=self.id)
#         )
#         self._execute_received_task(register_records_uids_task)
#
#         pubkey = party.recv(Task(ArbiterMethod.public_key, from_id=self.id, to_id=party.arbiter), recv_results=False) # TODO
#         self.save_pubkey(pubkey)
#
#         self.loop(batcher=self.batcher, communicator=party)
#
#         finalize_task = party.recv(Task(method_name=Method.finalize, from_id=self.master_id, to_id=self.id))
#         self._execute_received_task(finalize_task)
#         logger.info("Finished member %s" % self.id)
#
#     def loop(self, batcher: Batcher, party: PartyCommunicator):
#         """ Run main training loop on the VFL member.
#
#         :param batcher: Batcher for creating training batches.
#         :param communicator: Communicator instance used for communication between VFL agents.
#
#         :return: None
#         """
#         logger.info("Member %s: entering training loop" % self.id)
#
#         for titer in batcher:
#             iter_start_time = time.time()
#             predict_task = party.recv(Task(Method.partial_prediction), recv_results=False)
#             member_partial_preds = self.partial_prediction(titer, predict_task)
#
#             party.send(send_to_id=party.master, method_name=Method.partial_prediction,
#                        method_kwargs=member_partial_preds)
#
#             calculate_grad_enc_task = party.recv(Method.compute_gradient, recv_results=False)
#             gradient_enc = self.calculate_grad_enc(calculate_grad_enc_task)
#
#             upd_task = party.send(
#                 send_to_id=party.arbiter,
#                 task=Task(ArbiterMethod.compute_updates, method_kwargs=gradient_enc)
#             )
#             upd = party.recv(upd_task, recv_results=True)
#
#             self.update_weights(upd=upd)
#
#             # if self.report_train_metrics_iteration > 0 and titer.seq_num % self.report_train_metrics_iteration == 0:
#             #     logger.debug(
#             #         f"Member %s: train loop - calculating train metrics on iteration %s of epoch %s"
#             #         % (self.id, titer.seq_num, titer.epoch)
#             #     )
#             #     predict_task = communicator.recv(
#             #         Task(method_name=Method.predict, from_id=self.master_id, to_id=self.id)
#             #     )
#             #     predictions = self._execute_received_task(predict_task)
#             #     communicator.send(self.master_id, Method.predict, result=predictions)
#             #
#             # if self.report_test_metrics_iteration > 0 and titer.seq_num % self.report_test_metrics_iteration == 0:
#             #     logger.debug(
#             #         f"Member %s: train loop - calculating train metrics on iteration %s of epoch %s"
#             #         % (self.id, titer.seq_num, titer.epoch)
#             #     )
#             #     predict_task = communicator.recv(
#             #         Task(method_name=Method.predict, from_id=self.master_id, to_id=self.id)
#             #     )
#             #     predictions = self._execute_received_task(predict_task)
#             #     communicator.send(self.master_id, Method.predict, result=predictions)
#             #
#             self._iter_time.append((titer.seq_num, time.time() - iter_start_time))
#
#     @property
#     @abstractmethod
#     def batcher(self) -> Batcher:
#         """ Get the batcher for training.
#
#         :return: Batcher instance.
#         """
#         ...
#
#     @abstractmethod
#     def records_uids(self) -> List[str]:
#         """ Get the list of existing dataset unique identifiers.
#
#         :return: List of unique identifiers.
#         """
#         ...
#
#     @abstractmethod
#     def register_records_uids(self, uids: List[str]):
#         """ Register unique identifiers to be used.
#
#         :param uids: List of unique identifiers.
#         :return: None
#         """
#         ...
#
#     @abstractmethod
#     def initialize(self):
#         """ Initialize the party member. """
#         ...
#
#     @abstractmethod
#     def finalize(self):
#         """ Finalize the party member. """
#         ...
#
#     @abstractmethod
#     def update_weights(self, uids: RecordsBatch, upd: DataTensor):
#         """ Update model weights based on input features and target values.
#
#         :param uids: Batch of record unique identifiers.
#         :param upd: Updated model weights.
#         """
#         ...
#
#     @abstractmethod
#     def predict(self, uids: RecordsBatch, use_test: bool = False) -> DataTensor:
#         """ Make predictions using the initialized model.
#
#         :param uids: Batch of record unique identifiers.
#         :param use_test: Flag indicating whether to use the test data.
#
#         :return: Model predictions.
#         """
#         ...
#
#     @abstractmethod
#     def update_predict(self, upd: DataTensor, previous_batch: RecordsBatch, batch: RecordsBatch) -> DataTensor:
#         """ Update model weights and make predictions.
#
#         :param upd: Updated model weights.
#         :param previous_batch: Previous batch of record unique identifiers.
#         :param batch: Current batch of record unique identifiers.
#
#         :return: Model predictions.
#         """
#         ...
#
#     def save_pubkey(self, public_key: Keys):
#         self._public_key = public_key
