import collections
import itertools
import logging
import time
from abc import ABC, abstractmethod
from concurrent.futures import Future
from dataclasses import dataclass
from typing import Any, Iterator, List, Optional

import torch

logger = logging.getLogger(__name__)


DataTensor = torch.Tensor
# in reality, it will be a DataTensor but with one more dimension
PartyDataTensor = List[torch.Tensor]

RecordsBatch = List[str]


@dataclass(frozen=True)
class TrainingIteration:
    seq_num: int
    subiter_seq_num: int
    epoch: int
    batch: RecordsBatch
    previous_batch: Optional[RecordsBatch]
    participating_members: List[str]


class Batcher(ABC):
    # todo: add docs
    uids: List[str]

    @abstractmethod
    def __iter__(self) -> Iterator[TrainingIteration]:
        ...


class ParticipantFuture(Future):
    def __init__(self, participant_id: str):
        super().__init__()
        self.participant_id = participant_id


class PartyCommunicator(ABC):
    # todo: add docs
    world_size: int
    # todo: add docs about order guaranteeing
    members: List[str]

    @abstractmethod
    def randezvous(self):
        ...

    @property
    @abstractmethod
    def is_ready(self) -> bool:
        ...

    @abstractmethod
    def send(self, send_to_id: str, method_name: str, require_answer: bool = True, **kwargs) -> ParticipantFuture:
        ...

    # todo: shouldn't we replace it with message type?
    @abstractmethod
    def broadcast(
        self,
        method_name: str,
        mass_kwargs: Optional[List[Any]] = None,
        participating_members: Optional[List[str]] = None,
        parent_id: Optional[str] = None,
        require_answer: bool = True,
        include_current_participant: bool = False,
        **kwargs,
    ) -> List[ParticipantFuture]:
        ...

    @abstractmethod
    def run(self):
        ...


class Party(ABC):
    # todo: add docs
    world_size: int
    members: List[str]

    @abstractmethod
    def records_uids(self) -> List[List[str]]:
        ...

    @abstractmethod
    def register_records_uids(self, uids: List[str]):
        ...

    @abstractmethod
    def initialize(self):
        ...

    @abstractmethod
    def finalize(self):
        ...

    @abstractmethod
    def update_weights(self, upd: PartyDataTensor):
        ...

    @abstractmethod
    def predict(self, uids: List[str], use_test: bool = False) -> PartyDataTensor:
        ...

    @abstractmethod
    def update_predict(
        self, participating_members: List[str], batch: RecordsBatch, previous_batch: RecordsBatch, upd: PartyDataTensor
    ) -> PartyDataTensor:
        ...


class PartyMaster(ABC):
    # todo: add docs
    id: str
    epochs: int
    report_train_metrics_iteration: int
    report_test_metrics_iteration: int
    target: DataTensor
    target_uids: List[str]
    test_target: DataTensor

    _iter_time: list[tuple[int, float]] = list()

    @property
    def train_timings(self) -> list:
        return self._iter_time

    def run(self, party: Party):
        logger.info("Running master %s" % self.id)
        uids = self.synchronize_uids(party=party)

        self.master_initialize(party=party)

        self.loop(batcher=self.make_batcher(uids=uids, party=party), party=party)

        self.master_finalize(party=party)
        logger.info("Finished master %s" % self.id)

    def loop(self, batcher: Batcher, party: Party):
        logger.info("Master %s: entering training loop" % self.id)
        updates = self.make_init_updates(party.world_size)

        for titer in batcher:
            logger.debug(
                f"Master %s: train loop - starting batch %s (sub iter %s) on epoch %s"
                % (self.id, titer.seq_num, titer.subiter_seq_num, titer.epoch)
            )
            iter_start_time = time.time()

            party_predictions = party.update_predict(
                titer.participating_members, titer.batch, titer.previous_batch, updates
            )

            predictions = self.aggregate(titer.participating_members, party_predictions)

            updates = self.compute_updates(
                titer.participating_members, predictions, party_predictions, party.world_size, titer.subiter_seq_num
            )

            if self.report_train_metrics_iteration > 0 and titer.seq_num % self.report_train_metrics_iteration == 0:
                logger.debug(
                    f"Master %s: train loop - reporting train metrics on iteration %s of epoch %s"
                    % (self.id, titer.seq_num, titer.epoch)
                )
                party_predictions = party.predict(batcher.uids)
                predictions = self.aggregate(party.members, party_predictions, infer=True)
                self.report_metrics(self.target, predictions, name="Train")

            if self.report_test_metrics_iteration > 0 and titer.seq_num % self.report_test_metrics_iteration == 0:
                logger.debug(
                    f"Master %s: train loop - reporting test metrics on iteration %s of epoch %s"
                    % (self.id, titer.seq_num, titer.epoch)
                )
                party_predictions = party.predict(uids=batcher.uids, use_test=True)
                predictions = self.aggregate(party.members, party_predictions, infer=True)
                self.report_metrics(self.test_target, predictions, name="Test")

            self._iter_time.append((titer.seq_num, time.time() - iter_start_time))

    def synchronize_uids(self, party: Party) -> List[str]:
        logger.debug("Master %s: synchronizing uids for party of size %s" % (self.id, party.world_size))
        all_records_uids = party.records_uids()
        uids = itertools.chain(self.target_uids, (uid for member_uids in all_records_uids for uid in set(member_uids)))
        shared_uids = sorted([uid for uid, count in collections.Counter(uids).items() if count == party.world_size + 1])

        logger.debug("Master %s: registering shared uids f size %s" % (self.id, len(shared_uids)))
        party.register_records_uids(shared_uids)

        set_shared_uids = set(shared_uids)
        uid2idx = {uid: i for i, uid in enumerate(self.target_uids) if uid in set_shared_uids}
        selected_tensor_idx = [uid2idx[uid] for uid in shared_uids]

        self.target = self.target[selected_tensor_idx]
        self.target_uids = shared_uids

        logger.debug("Master %s: record uids has been successfully synchronized")

        return shared_uids

    @abstractmethod
    def make_batcher(self, uids: List[str], party: Party) -> Batcher:
        ...

    @abstractmethod
    def master_initialize(self, party: Party):
        ...

    @abstractmethod
    def master_finalize(self, party: Party):
        ...

    @abstractmethod
    def make_init_updates(self, world_size: int) -> PartyDataTensor:
        ...

    @abstractmethod
    def aggregate(
        self, participating_members: List[str], party_predictions: PartyDataTensor, infer: bool = False
    ) -> DataTensor:
        ...

    @abstractmethod
    def compute_updates(
        self,
        participating_members: List[str],
        predictions: DataTensor,
        party_predictions: PartyDataTensor,
        world_size: int,
        subiter_seq_num: int,
    ) -> List[DataTensor]:
        ...

    @abstractmethod
    def report_metrics(self, y: DataTensor, predictions: DataTensor, name: str):
        ...


class PartyMember(ABC):
    # todo: add docs
    id: str

    @abstractmethod
    def records_uids(self) -> List[str]:
        ...

    @abstractmethod
    def register_records_uids(self, uids: List[str]):
        ...

    @abstractmethod
    def initialize(self):
        ...

    @abstractmethod
    def finalize(self):
        ...

    @abstractmethod
    def update_weights(self, uids: RecordsBatch, upd: DataTensor):
        ...

    @abstractmethod
    def predict(self, uids: RecordsBatch, use_test: bool = False) -> DataTensor:
        ...

    @abstractmethod
    def update_predict(self, upd: DataTensor, previous_batch: RecordsBatch, batch: RecordsBatch) -> DataTensor:
        ...
