from abc import ABC, abstractmethod
from typing import List

from stalactite.base import Batcher
from stalactite.communications import Party


class PartyMaster(ABC):
    epochs: int
    batcher: Batcher

    def run(self):
        party = self.randezvous()
        uids = self.synchronize_uids(party)

        self.master_initialize()

        self.loop(uids, party)

        self.master_finalize()

    @abstractmethod
    def randezvous(self) -> Party:
        ...

    @abstractmethod
    def synchronize_uids(self, party: Party) -> List[str]:
        ...

    def loop(self, batcher: Batcher, party: Party):
        for epoch in range(self.epochs):
            for batch in batcher:
                pass

        ...

    @abstractmethod
    def master_initialize(self):
        ...

    @abstractmethod
    def master_finalize(self):
        ...
