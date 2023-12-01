import torch

from party_master_Impl import PartyMasterImpl
from party_member_Impl import PartyMemberImpl
from party_Impl import PartyImpl


if __name__ == "__main__":

    master = PartyMasterImpl(
        epochs=1,
        report_train_metrics_iteration=5,
        report_test_metrics_iteration=5,
        Y=torch.randint(0, 2, (5,))
    )
    members = [PartyMemberImpl() for _ in range(3)]
    party = PartyImpl(master, members)
    party.initialize()
    party.run()
