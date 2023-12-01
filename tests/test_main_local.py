import torch

from stalactite.party_master_Impl import PartyMasterImpl
from stalactite.party_member_Impl import PartyMemberImpl
from stalactite.party_Impl import PartyImpl


def test_one():
    x = 5
    assert 1 == 1


def test_integration_local_party():
    members_count = 3
    epochs = 1
    batch_size = 2
    ds_rows = 5
    batches = epochs * (ds_rows//batch_size+1)
    rhs_send = members_count * batches

    master = PartyMasterImpl(
        epochs=epochs,
        report_train_metrics_iteration=5,
        report_test_metrics_iteration=5,
        Y=torch.randint(0, 2, (5,))
    )
    members = [PartyMemberImpl() for _ in range(members_count)]
    party = PartyImpl(master, members)
    party.initialize()
    party.run()
    assert 1 == 1
    assert master.epoch_counter == epochs
    assert master.batch_counter == batches
    assert party.party_counter["rhs_send"] == rhs_send
