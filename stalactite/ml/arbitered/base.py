import enum
import logging
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, List, Optional, Union, TypeVar

import numpy as np
import torch

from stalactite.base import (
    PartyMember,
    PartyMaster,
    PartyCommunicator,
    Method, ArbiteredMethod,
    MethodKwargs,
    Task,
    DataTensor,
    RecordsBatch,
    Batcher,
    PartyAgent, PartyDataTensor,
)
from stalactite.batching import ListBatcher

logger = logging.getLogger(__name__)

T = TypeVar('T', np.ndarray, torch.Tensor)


@dataclass
class Keys:
    """ Helper class for generated public and private HE key pair. """
    public: Any = None
    private: Any = None


class Role(str, enum.Enum):
    arbiter = "arbiter"
    master = "master"
    member = "member"


class SecurityProtocol(ABC):
    """ Base proxy class for Homomorphic Encryption (HE) protocol. """
    _keys: Optional[Keys] = None

    @abstractmethod
    def initialize(self):
        """ Initialize protocol if requires some additional attributes after kays are added. """
        ...

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

    @abstractmethod
    def multiply_plain_cypher(self, plain_arr: Any, cypher_arr: T) -> T:
        """ Multiply plaintext matrix to the encrypted matrix.

        :param plain_arr: Plaintext matrix of shape (n; x).
        :param cypher_arr: Encrypted matrix of shape (x; m).

        :return: Encrypted matrix - result of the matmul operation of shape (n; m).
        """
        ...

    @abstractmethod
    def add_matrices(self, array1: Union[Any, T], array2: T) -> T:
        """ Add plain or cyphered matrix to cyphered matrix.

        :param array1: Plaintext or encrypted matrix of shape (n; m).
        :param array2: Encrypted matrix of shape (n; m).

        :return: Encrypted matrix - result of the matrix summation of shape (n; m).
        """
        ...

    @abstractmethod
    def encode(self, array: [Any]) -> T:
        """ Encode plain tensor for multiplication.
        If the multiplication is perfomed on encoded and cyphered data, return encoded data, otherwise,
        return initial data without transformation.

        :param array: Plaintext matrix.

        :return: Encoded or plaintext matrix.
        """
        ...


class SecurityProtocolArbiter(SecurityProtocol, ABC):
    """ Expanded functionality (trusted party arbiter) base proxy class for Homomorphic Encryption (HE) protocol. """

    @abstractmethod
    def generate_keys(self):
        """ Generate private and public keys. """
        ...

    @abstractmethod
    def decrypt(self, encrypted_data: Any) -> torch.Tensor:
        """ Decrypt data using private key.

        :param encrypted_data: Data to decrypt using the private key.
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

    def get_public_key(self) -> Keys:
        if self.security_protocol is not None:
            return self.security_protocol.public_key
        return Keys()

    def _collect_gradients_for_update(self, master_task: Task, member_tasks: List[Task]) -> dict:
        """ Collect gradients from agents, create dictionary of sender-gradient pairs. """
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
        """ Run main arbiter loops (training and inference). """
        logger.info("Running arbiter %s" % self.id)
        if self.do_train:
            self.fit(party)

        if self.do_predict:
            self.inference(party)

    def fit(self, party: PartyCommunicator):
        """ Run arbiter for VFL training. """
        initialize_task = party.recv(Task(method_name=Method.initialize, from_id=party.master, to_id=self.id))
        self.execute_received_task(initialize_task)

        register_records_uids_task = party.recv(
            Task(method_name=Method.register_records_uids, from_id=party.master, to_id=self.id)
        )
        self.execute_received_task(register_records_uids_task)

        register_test_records_uids_task = party.recv(
            Task(method_name=Method.register_records_uids, from_id=party.master, to_id=self.id)
        )
        self.execute_received_task(register_test_records_uids_task)

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

        self.loop(batcher=self.make_batcher(), party=party)

        finalize_task = party.recv(Task(method_name=Method.finalize, from_id=party.master, to_id=self.id))
        self.execute_received_task(finalize_task)
        logger.info("Finished master %s" % self.id)

    def loop(self, batcher: Batcher, party: PartyCommunicator) -> None:
        """ Main training loop on arbiter. """
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

    def inference(self, party: PartyCommunicator):
        """ Run VFL inference on arbiter. """
        logger.info('Arbiter is not included in the inference loop.')

    def inference_loop(self, batcher: Batcher, party: PartyCommunicator) -> None:
        """ Run VFL inference loop on arbiter. """
        pass


class ArbiteredPartyMaster(PartyMaster, PartyMember, ABC):
    security_protocol: Optional[SecurityProtocol] = None
    is_initialized: bool
    is_finalized: bool
    _batch_size: int
    _eval_batch_size: int
    _batcher: Batcher

    @abstractmethod
    def predict(
            self, uids: Optional[List[str]], is_infer: bool = False  # , batch: Any = None
    ) -> tuple[DataTensor, DataTensor]:
        """ Calculate predictions on current model.

        :param uids: Batch of record unique identifiers.

        :return: Predictions, target.
        """
        ...

    @abstractmethod
    def predict_partial(self, uids: RecordsBatch) -> DataTensor:  # , batch: Any
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
        """ Run main master loops (training and inference). """

        if self.do_train:
            self.fit(party)

        if self.do_predict:
            self.inference(party)

    def fit(self, party: PartyCommunicator):
        """ Run master for VFL training. """

        logger.info("Running master %s" % self.id)

        records_uids_tasks = party.broadcast(
            ArbiteredMethod.records_uids,
            participating_members=party.members,
        )
        records_uids_results = party.gather(records_uids_tasks, recv_results=True)

        collected_uids_results = [task.result for task in records_uids_results]

        test_records_uids_tasks = party.broadcast(
            ArbiteredMethod.records_uids,
            participating_members=party.members,
            method_kwargs=MethodKwargs(other_kwargs={'is_infer': True}),
        )

        test_records_uids_results = party.gather(test_records_uids_tasks, recv_results=True)

        test_collected_uids_results = [task.result for task in test_records_uids_results]

        party.broadcast(
            ArbiteredMethod.initialize,
            participating_members=party.members + [party.arbiter],
        )

        self.initialize(is_infer=False)

        uids = self.synchronize_uids(collected_uids_results, world_size=party.world_size)
        test_uids = self.synchronize_uids(test_collected_uids_results, world_size=party.world_size, is_infer=True)

        party.broadcast(
            ArbiteredMethod.register_records_uids,
            method_kwargs=MethodKwargs(other_kwargs={"uids": uids}),
            participating_members=party.members + [party.master] + [party.arbiter],
        )

        register_records_uids_task = party.recv(
            Task(method_name=Method.register_records_uids, from_id=party.master, to_id=self.id)
        )

        self.execute_received_task(register_records_uids_task)

        party.broadcast(
            ArbiteredMethod.register_records_uids,
            method_kwargs=MethodKwargs(other_kwargs={"uids": test_uids, "is_infer": True}),
            participating_members=party.members + [party.master] + [party.arbiter],
        )

        register_test_records_uids_task = party.recv(
            Task(method_name=Method.register_records_uids, from_id=party.master, to_id=self.id)
        )

        self.execute_received_task(register_test_records_uids_task)

        pk_task = party.send(method_name=ArbiteredMethod.get_public_key, send_to_id=party.arbiter)
        pk = party.recv(pk_task, recv_results=True).result

        if self.security_protocol is not None:
            self.security_protocol.keys = pk
            self.security_protocol.initialize()

        self.loop(batcher=self.make_batcher(uids=uids, party_members=party.members), party=party)

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

            if self.report_train_metrics_iteration > 0 and titer.seq_num % self.report_train_metrics_iteration == 0:
                predict_tasks = party.broadcast(
                    ArbiteredMethod.predict,
                    method_kwargs=MethodKwargs(
                        other_kwargs={'uids': None, 'is_infer': False}
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

            if self.report_test_metrics_iteration > 0 and titer.seq_num % self.report_test_metrics_iteration == 0:
                predict_tasks = party.broadcast(
                    ArbiteredMethod.predict,
                    method_kwargs=MethodKwargs(
                        other_kwargs={'uids': None, 'is_infer': True}
                    ),
                    participating_members=titer.participating_members
                )

                master_predictions, targets = self.predict(uids=None, is_infer=True)
                participant_partial_predictions_tasks = party.gather(predict_tasks, recv_results=True)
                aggr_predictions = self.aggregate_predictions(
                    master_predictions=master_predictions,
                    members_predictions=[task.result for task in participant_partial_predictions_tasks]
                )

                self.report_metrics(targets, aggr_predictions, 'Test', step=titer.seq_num)

            self._iter_time.append((titer.seq_num, time.time() - iter_start_time))

    def inference(self, party: PartyCommunicator):
        """ Run VFL inference on master. """
        if not self.do_train:
            records_uids_tasks = party.broadcast(
                Method.records_uids,
                participating_members=party.members,
                method_kwargs=MethodKwargs(other_kwargs={'is_infer': True})
            )
            records_uids_results = party.gather(records_uids_tasks, recv_results=True)

            collected_uids_results = [task.result for task in records_uids_results]

        party.broadcast(
            Method.initialize,
            participating_members=party.members,
            method_kwargs=MethodKwargs(other_kwargs={'is_infer': True})

        )
        self.initialize(is_infer=True)
        if not self.do_train:
            uids = self.synchronize_uids(collected_uids_results, world_size=party.world_size, is_infer=True)
            party.broadcast(
                Method.register_records_uids,
                method_kwargs=MethodKwargs(other_kwargs={"uids": uids, 'is_infer': True}),
                participating_members=party.members,
            )
        else:
            uids = None
        self.inference_loop(
            batcher=self.make_batcher(uids=uids, party_members=party.members, is_infer=True),
            party=party
        )

        party.broadcast(
            Method.finalize,
            participating_members=party.members,
            method_kwargs=MethodKwargs(other_kwargs={'is_infer': True})
        )
        self.finalize(is_infer=True)
        logger.info("Finished master %s" % self.id)

    def inference_loop(self, batcher: Batcher, party: PartyCommunicator) -> None:
        logger.info("Master %s: entering inference loop" % self.id)
        party_predictions_test = defaultdict(list)
        test_targets = torch.tensor([])
        for titer in batcher:
            if titer.last_batch:
                break
            logger.debug(
                f"Master %s: inference loop - starting batch %s (sub iter %s) on epoch %s"
                % (self.id, titer.seq_num, titer.subiter_seq_num, titer.epoch)
            )
            iter_start_time = time.time()
            predict_test_tasks = party.broadcast(
                Method.predict,
                method_kwargs=MethodKwargs(other_kwargs={"uids": titer.batch, "is_infer": True}),
                participating_members=titer.participating_members,
            )
            predictions_master, target = self.predict(uids=titer.batch, is_infer=True)
            party_predictions_test[self.id].append(predictions_master)
            for task in party.gather(predict_test_tasks, recv_results=True):
                party_predictions_test[task.from_id].append(task.result)
            test_targets = torch.cat([test_targets, target])
            self._iter_time.append((titer.seq_num, time.time() - iter_start_time))
        aggr_party_predictions = self._aggregate_batched_predictions(party.members, party_predictions_test)
        predictions = self.aggregate_predictions(
            torch.cat(party_predictions_test[self.id], dim=1),
            aggr_party_predictions
        )
        self.report_metrics(test_targets, predictions, name="Test", step=-1)

    def _aggregate_batched_predictions(
            self,
            party_members: List[str],
            batched_party_predictions: dict[str, List[DataTensor]]
    ):
        return [torch.cat(batched_party_predictions[member], dim=1) for member in party_members]


class ArbiteredPartyMember(PartyMember, ABC):
    security_protocol: Optional[SecurityProtocol] = None
    master: str

    @abstractmethod
    def predict(self, uids: Optional[List[str]], is_infer: bool = False) -> DataTensor:
        """ Calculate predictions on current model.

        :param uids: Batch of record unique identifiers.
        :param is_infer: Whether to use test split.

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
        """ Run main member loops (training and inference). """

        logger.info("Running member %s" % self.id)

        if self.do_train:
            self.fit(party)

        if self.do_predict:
            self.inference(party)

    def inference(self, party: PartyCommunicator):
        """ Run VFL inference on member. """
        if not self.do_train:
            synchronize_uids_task = party.recv(
                Task(method_name=Method.records_uids, from_id=self.master, to_id=self.id)
            )
            uids = self.execute_received_task(synchronize_uids_task)
            party.send(self.master, Method.records_uids, result=uids)

        initialize_task = party.recv(Task(method_name=Method.initialize, from_id=self.master, to_id=self.id))
        self.execute_received_task(initialize_task)
        if not self.do_train:
            register_records_uids_task = party.recv(
                Task(method_name=Method.register_records_uids, from_id=self.master, to_id=self.id)
            )
            self.execute_received_task(register_records_uids_task)

        self.inference_loop(batcher=self.make_batcher(is_infer=True), party=party)

        finalize_task = party.recv(Task(method_name=Method.finalize, from_id=self.master, to_id=self.id))
        self.execute_received_task(finalize_task)
        logger.info("Finished member %s" % self.id)

    def inference_loop(self, batcher: Batcher, party: PartyCommunicator) -> None:
        logger.info("Member %s: entering inference loop" % self.id)

        for titer in batcher:
            if titer.last_batch:
                break
            if titer.participating_members is not None:
                if self.id not in titer.participating_members:
                    logger.debug(f'Member {self.id} skipping {titer.seq_num}.')
                    continue

            predict_task = party.recv(Task(method_name=Method.predict, from_id=self.master, to_id=self.id))
            predictions = self.execute_received_task(predict_task)
            party.send(self.master, Method.predict, result=predictions)

    def fit(self, party: PartyCommunicator):
        """ Run member for VFL training. """

        synchronize_uids_task = party.recv(
            Task(method_name=Method.records_uids, from_id=party.master, to_id=self.id)
        )
        uids = self.execute_received_task(synchronize_uids_task)
        party.send(party.master, Method.records_uids, result=uids)

        synchronize_test_uids_task = party.recv(
            Task(method_name=Method.records_uids, from_id=self.master, to_id=self.id)
        )
        test_uids = self.execute_received_task(synchronize_test_uids_task)
        party.send(self.master, Method.records_uids, result=test_uids)

        initialize_task = party.recv(Task(method_name=Method.initialize, from_id=party.master, to_id=self.id))
        self.execute_received_task(initialize_task)

        register_records_uids_task = party.recv(
            Task(method_name=Method.register_records_uids, from_id=party.master, to_id=self.id)
        )

        self.execute_received_task(register_records_uids_task)

        register_test_records_uids_task = party.recv(
            Task(method_name=Method.register_records_uids, from_id=self.master, to_id=self.id)
        )
        self.execute_received_task(register_test_records_uids_task)

        pk_task = party.send(method_name=ArbiteredMethod.get_public_key, send_to_id=party.arbiter)
        pk = party.recv(pk_task, recv_results=True).result

        if self.security_protocol is not None:
            self.security_protocol.keys = pk
            self.security_protocol.initialize()

        self.loop(batcher=self.make_batcher(), party=party)

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

            if self.report_train_metrics_iteration > 0 and titer.seq_num % self.report_train_metrics_iteration == 0:
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
            if self.report_test_metrics_iteration > 0 and titer.seq_num % self.report_test_metrics_iteration == 0:
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
