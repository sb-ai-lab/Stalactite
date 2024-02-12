from stalactite.base import Batcher, PartyCommunicator
from stalactite.ml.arbitered.base import ArbiteredPartyMember


class ArbiteredPartyMemberLinReg(ArbiteredPartyMember):
    def loop(self, batcher: Batcher, party: PartyCommunicator) -> None:
        ...
