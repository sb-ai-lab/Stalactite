import collections
import itertools
import logging
from abc import ABC, abstractmethod
from concurrent.futures import Future
from typing import List, Any, Optional

import torch

logger = logging.getLogger(__name__)


DataTensor = torch.Tensor
# in reality, it will be a DataTensor but with one more dimension
PartyDataTensor = List[torch.Tensor]


class Batcher(ABC):
    # todo: add docs
    uids: List[str]

    @abstractmethod
    def __iter__(self):
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
    def send(self, send_to_id: str,  method_name: str, require_answer: bool = True, **kwargs) -> ParticipantFuture:
        ...

    # todo: shouldn't we replace it with message type?
    @abstractmethod
    def broadcast(self,
                  method_name: str,
                  mass_kwargs: Optional[List[Any]] = None,
                  parent_id: Optional[str] = None,
                  require_answer: bool = True,
                  include_current_participant: bool = False,
                  **kwargs) -> List[ParticipantFuture]:
        ...

    @abstractmethod
    def run(self):
        ...


class Party(ABC):
    # todo: add docs
    world_size: int

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
    def predict(self, use_test: bool = False) -> PartyDataTensor:
        ...

    @abstractmethod
    def update_predict(self, batch: List[str], upd: PartyDataTensor) -> PartyDataTensor:
        ...


class PartyMaster(ABC):
    # todo: add docs
    id: str
    epochs: int
    report_train_metrics_iteration: int
    report_test_metrics_iteration: int
    target: DataTensor
    target_uids: List[str]

    def run(self, party: Party):
        logger.info("Running master %s" % self.id)
        uids = self.synchronize_uids(party=party)

        self.master_initialize(party=party)

        self.loop(
            batcher=self.make_batcher(uids=uids),
            party=party
        )

        self.master_finalize(party=party)
        logger.info("Finished master %s" % self.id)

    def loop(self, batcher: Batcher, party: Party):
        logger.info("Master %s: entering training loop" % self.id)
        updates = self.make_init_updates(party.world_size)
        for epoch in range(self.epochs):
            logger.debug(f"Master %s: train loop - starting EPOCH %s / %s" % (self.id, epoch, self.epochs))
            for i, batch in enumerate(batcher):
                logger.debug(f"Master %s: train loop - starting batch %s on epoch %s" % (self.id, i, epoch))
                party_predictions = party.update_predict(batch, updates)
                predictions = self.aggregate(party_predictions)
                updates = self.compute_updates(predictions, party_predictions, party.world_size)

                if self.report_train_metrics_iteration > 0 and i % self.report_train_metrics_iteration == 0:
                    logger.debug(f"Master %s: train loop - reporting train metrics on iteration %s of epoch %s"
                                 % (self.id, i, epoch))
                    party_predictions = party.predict()
                    predictions = self.aggregate(party_predictions)
                    self.report_metrics(self.target, predictions, name="Train")

                if self.report_test_metrics_iteration > 0 and i % self.report_test_metrics_iteration == 0:
                    logger.debug(f"Master %s: train loop - reporting test metrics on iteration %s of epoch %s"
                                 % (self.id, i, epoch))
                    party_predictions = party.predict(use_test=True)
                    predictions = self.aggregate(party_predictions)
                    self.report_metrics(self.target, predictions, name="Test")

    def synchronize_uids(self, party: Party) -> List[str]:
        logger.debug("Master %s: synchronizing uids for party of size %s" % (self.id, party.world_size))
        all_records_uids = party.records_uids()
        uids = itertools.chain(
            self.target_uids,
            (uid for member_uids in all_records_uids for uid in set(member_uids))
        )
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
    def make_batcher(self, uids: List[str]) -> Batcher:
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
    def aggregate(self, party_predictions: PartyDataTensor) -> DataTensor:
        ...

    @abstractmethod
    def compute_updates(self, predictions: DataTensor, party_predictions: PartyDataTensor, world_size: int) \
            -> List[DataTensor]:
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
    def update_weights(self, upd: DataTensor):
        ...

    @abstractmethod
    def predict(self, batch: List[str]) -> DataTensor:
        ...

    @abstractmethod
    def update_predict(self, batch: List[str], upd: DataTensor) -> DataTensor:
        ...
