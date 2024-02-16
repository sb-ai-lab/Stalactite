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
    PartyAgent, PartyDataTensor,
)
from stalactite.batching import ListBatcher

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


class ArbiteredMethod(str, enum.Enum): # TODO Move to method
    service_return_answer = "service_return_answer"
    service_heartbeat = "service_heartbeat"

    records_uids = "records_uids"
    register_records_uids = "register_records_uids"

    initialize = "initialize"
    finalize = "finalize"

    update_weights = "update_weights"
    predict = "predict"
    update_predict = "update_predict"

    get_public_key = "get_public_key"
    predict_partial = "predict_partial"
    compute_gradient = "compute_gradient"
    calculate_updates = "calculate_updates"


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

    @abstractmethod
    def register_records_uids(self, uids: List[str]) -> None:
        """ Register unique identifiers to be used.

        :param uids: List of unique identifiers.
        :return: None
        """
        ...

    @property
    @abstractmethod
    def batcher(self) -> Batcher:
        ...

    def get_public_key(self) -> Keys:
        return self.security_protocol.public_key

    def _collect_gradients_for_update(self, master_task: Task, member_tasks: List[Task]) -> dict:
        gradients = {master_task.from_id: master_task.kwargs_dict['gradient']}
        for task in member_tasks:
            gradients[task.from_id] = task.kwargs_dict['gradient']

        return gradients

    @abstractmethod
    def calculate_updates(self, gradients: dict) -> dict[str, DataTensor]:
        """ Using local encrypted gradients, calculate updates on the main model for each agent.

        :param gradients: Collected dict of agent_id: local_gradient key-value pairs
        :return: Dictionary of agent_id: local_model_update  key-value pairs.
        """
        ...

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
            result=keys,
            participating_members=party.members + [party.master]
        )

        self.loop(batcher=self.batcher, party=party)

        finalize_task = party.recv(Task(method_name=Method.finalize, from_id=party.master, to_id=self.id))
        self.execute_received_task(finalize_task)
        logger.info("Finished master %s" % self.id)

    def loop(self, batcher: Batcher, party: PartyCommunicator) -> None:
        logger.info("Arbiter %s: entering training loop" % self.id)
        for titer in batcher:
            master_gradient = party.recv(
                Task(method_name=ArbiteredMethod.calculate_updates, from_id=party.master, to_id=self.id),
                recv_results=False
            )

            members_tasks = [
                Task(method_name=ArbiteredMethod.calculate_updates, from_id=member_id, to_id=self.id)
                for member_id in titer.participating_members
            ]
            members_gradients = party.gather(members_tasks, recv_results=False)

            grads = self._collect_gradients_for_update(master_task=master_gradient, member_tasks=members_gradients)
            models_updates = self.calculate_updates(grads)

            results, agents = [], []
            for agent_id, upd in models_updates.items():
                results.append(upd)
                agents.append(agent_id)

            party.scatter(
                method_name=ArbiteredMethod.calculate_updates,
                result=results,
                participating_members=agents
            )


class ArbiteredPartyMaster(PartyMaster, PartyMember, ABC):
    security_protocol: Optional[SecurityProtocol] = None
    is_initialized: bool
    is_finalized: bool
    _batch_size: int
    _batcher: Batcher

    def make_batcher(self, uids: List[str], party_members: List[str]) -> Batcher:
        logger.info("Master %s: making a batcher for uids of length %s" % (self.id, len(uids)))
        self.check_if_ready()
        assert party_members is not None, "Master is trying to initialize batcher without members list"
        batcher = ListBatcher(epochs=self.epochs, members=party_members, uids=uids, batch_size=self._batch_size)
        self._batcher = batcher
        return batcher

    @abstractmethod
    def predict(self, uids: Optional[List[str]], is_test: bool = False) -> tuple[DataTensor, DataTensor]:
        """ Calculate predictions on current model.

        :param uids: Batch of record unique identifiers.

        :return: Predictions, target.
        """
        ...

    @abstractmethod
    def predict_partial(self, uids: RecordsBatch) -> DataTensor:
        """ Calculate error using the current model.

        :param uids: Batch of record unique identifiers.

        :return: Difference between the master predictions and the target.
        """
        ...

    @abstractmethod
    def aggregate_partial_predictions(
            self, master_prediction: DataTensor, members_predictions: PartyDataTensor, uids: RecordsBatch
    ) -> DataTensor:
        """ Calculate main model error using the current model.

        :param master_prediction: Error calculated on the master (XW - y).
        :param members_predictions: Predictions of the members.
        :param uids: Batch uids.

        :return: Difference between the aggregated predictions and the target.
        """
        ...

    @abstractmethod
    def compute_gradient(
            self,
            aggregated_predictions_diff: DataTensor,
            uids: List[str],
    ) -> DataTensor:
        """ Calculate local gradient.

        :param aggregated_predictions_diff: Difference between the aggregated predictions and the target.
        :param uids: Batch uids.

        :return: Local model gradient.
        """
        ...

    @abstractmethod
    def aggregate_predictions(
            self, master_predictions: DataTensor, members_predictions: PartyDataTensor,
    ) -> DataTensor:
        """ Aggregate master and members predictions .

        :param master_predictions: predictions made on master.
        :param members_predictions: Collected from members predictions.

        :return: Aggregated predictions.
        """
        ...

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

        self.security_protocol.keys = pk


        self.loop(batcher=self.make_batcher(uids=uids, party_members=party.members), party=party)
        #
        party.broadcast(
            Method.finalize,
            participating_members=party.members + [party.arbiter],
        )
        self.finalize()
        logger.info("Finished master %s" % self.id)

    def loop(self, batcher: Batcher, party: PartyCommunicator) -> None:
        """ Run main training loop on the VFL master.

        :param batcher: Batcher for creating training batches.
        :param communicator: Communicator instance used for communication between VFL agents.

        :return: None
        """
        logger.info("Master %s: entering training loop" % self.id)

        for titer in batcher:
            iter_start_time = time.time()
            participant_partial_pred_tasks = party.broadcast(
                ArbiteredMethod.predict_partial,
                participating_members=titer.participating_members,
                method_kwargs=MethodKwargs(
                    other_kwargs={'uids': titer.batch}
                )
            )

            master_partial_preds = self.predict_partial(uids=titer.batch)  # d

            participant_partial_predictions_tasks = party.gather(participant_partial_pred_tasks, recv_results=True)
            partial_predictions = [task.result for task in participant_partial_predictions_tasks]


            predictions_delta = self.aggregate_partial_predictions(
                master_prediction=master_partial_preds,
                members_predictions=partial_predictions,
                uids=titer.batch,
            )  # d


            party.broadcast(
                ArbiteredMethod.compute_gradient,
                method_kwargs=MethodKwargs(
                    tensor_kwargs={'aggregated_predictions_diff': predictions_delta},
                    other_kwargs={'uids': titer.batch}
                ),
                participating_members=titer.participating_members
            )

            master_gradient = self.compute_gradient(predictions_delta, titer.batch)  # g_enc
            calculate_updates_task = party.send(
                send_to_id=party.arbiter,
                method_name=ArbiteredMethod.calculate_updates,
                method_kwargs=MethodKwargs(tensor_kwargs={'gradient': master_gradient}),
            )
            model_updates = party.recv(calculate_updates_task, recv_results=True)

            self.update_weights(upd=model_updates.result, uids=titer.batch)
            # self.update_weights(upd=master_gradient, uids=titer.batch) # !!! TODO rm

            # TODO REPORT METRICS
            predict_tasks = party.broadcast(
                ArbiteredMethod.predict,
                method_kwargs=MethodKwargs(
                    other_kwargs={'uids': None, 'is_test': False}
                ),
                participating_members=titer.participating_members
            )

            master_predictions, targets = self.predict(uids=None)
            participant_partial_predictions_tasks = party.gather(predict_tasks, recv_results=True)
            aggr_predictions = self.aggregate_predictions(
                master_predictions=master_predictions,
                members_predictions=[task.result for task in participant_partial_predictions_tasks],
            )

            self.report_metrics(targets, aggr_predictions, 'Train', step=titer.seq_num)

            predict_tasks = party.broadcast(
                ArbiteredMethod.predict,
                method_kwargs=MethodKwargs(
                    other_kwargs={'uids': None, 'is_test': True}
                ),
                participating_members=titer.participating_members
            )

            master_predictions, targets = self.predict(uids=None, is_test=True)
            participant_partial_predictions_tasks = party.gather(predict_tasks, recv_results=True)
            aggr_predictions = self.aggregate_predictions(
                master_predictions=master_predictions,
                members_predictions=[task.result for task in participant_partial_predictions_tasks]
            )

            self.report_metrics(targets, aggr_predictions, 'Test', step=titer.seq_num)

            self._iter_time.append((titer.seq_num, time.time() - iter_start_time))


class ArbiteredPartyMember(PartyMember, ABC):
    security_protocol: Optional[SecurityProtocol] = None

    @abstractmethod
    def predict(self, uids: Optional[List[str]], is_test: bool = False) -> DataTensor:
        """ Calculate predictions on current model.

        :param uids: Batch of record unique identifiers.

        :return: Predictions.
        """
        ...


    @abstractmethod
    def predict_partial(self, uids: RecordsBatch) -> DataTensor:
        """ Calculate error using the current model.

        :param uids: Batch of record unique identifiers.

        :return: Predictions of the local model.
        """
        ...

    @abstractmethod
    def compute_gradient(
            self,
            aggregated_predictions_diff: DataTensor,
            uids: List[str],
    ) -> DataTensor:
        """ Calculate local gradient.

        :param aggregated_predictions_diff: Difference between the aggregated predictions and the target.
        :param uids: Batch uids.

        :return: Local model gradient.
        """
        ...

    def run(self, party: PartyCommunicator) -> None:
        logger.info("Running member %s" % self.id)

        synchronize_uids_task = party.recv(
            Task(method_name=Method.records_uids, from_id=party.master, to_id=self.id)
        )
        uids = self.execute_received_task(synchronize_uids_task)
        party.send(party.master, Method.records_uids, result=uids)

        initialize_task = party.recv(Task(method_name=Method.initialize, from_id=party.master, to_id=self.id))
        self.execute_received_task(initialize_task)

        register_records_uids_task = party.recv(
            Task(method_name=Method.register_records_uids, from_id=party.master, to_id=self.id)
        )

        self.execute_received_task(register_records_uids_task)

        pk_task = party.send(method_name=ArbiteredMethod.get_public_key,
                             send_to_id=party.arbiter)  # TODO move to execute_task
        pk = party.recv(pk_task, recv_results=True).result

        self.security_protocol.keys = pk

        self.loop(batcher=self.batcher, party=party)

        finalize_task = party.recv(Task(method_name=Method.finalize, from_id=party.master, to_id=self.id))
        self.execute_received_task(finalize_task)
        logger.info("Finished member %s" % self.id)

    def loop(self, batcher: Batcher, party: PartyCommunicator) -> None:
        for titer in batcher:
            iter_start_time = time.time()
            participant_partial_pred_task = party.recv(
                Task(ArbiteredMethod.predict_partial, from_id=party.master, to_id=self.id),
                recv_results=False
            )
            member_partial_preds = self.execute_received_task(participant_partial_pred_task)

            party.send(
                send_to_id=party.master,
                method_name=ArbiteredMethod.predict_partial,
                result=member_partial_preds
            )

            compute_gradient_task = party.recv(
                Task(ArbiteredMethod.compute_gradient, from_id=party.master, to_id=self.id),
                recv_results=False
            )
            member_gradient = self.execute_received_task(compute_gradient_task)

            calculate_updates_task = party.send(
                send_to_id=party.arbiter,
                method_name=ArbiteredMethod.calculate_updates,
                method_kwargs=MethodKwargs(tensor_kwargs={'gradient': member_gradient}),
            )
            model_updates = party.recv(calculate_updates_task, recv_results=True)

            self.update_weights(upd=model_updates.result, uids=titer.batch)
            # self.update_weights(upd=member_gradient, uids=titer.batch) # !!! TODO rm

            # TODO REPORT METRICS
            predict_task = party.recv(
                Task(ArbiteredMethod.predict, from_id=party.master, to_id=self.id),
                recv_results=False
            )
            member_prediction = self.execute_received_task(predict_task)
            party.send(
                send_to_id=party.master,
                method_name=ArbiteredMethod.predict,
                result=member_prediction
            )

            predict_task = party.recv(
                Task(ArbiteredMethod.predict, from_id=party.master, to_id=self.id),
                recv_results=False
            )
            member_prediction = self.execute_received_task(predict_task)
            party.send(
                send_to_id=party.master,
                method_name=ArbiteredMethod.predict,
                result=member_prediction
            )

            self._iter_time.append((titer.seq_num, time.time() - iter_start_time))


