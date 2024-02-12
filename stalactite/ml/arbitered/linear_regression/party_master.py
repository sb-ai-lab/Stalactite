from stalactite.base import Batcher, PartyCommunicator
from stalactite.ml.arbitered.base import ArbiteredPartyMaster


class ArbiteredPartyMasterLinReg(ArbiteredPartyMaster):
    def loop(self, batcher: Batcher, party: PartyCommunicator) -> None:
        ...