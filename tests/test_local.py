import math
from threading import Thread

import torch

from stalactite.base import PartyMember
from stalactite.communications.local import LocalMasterPartyCommunicator, LocalMemberPartyCommunicator
from stalactite.mocks import MockPartyMasterImpl, MockPartyMemberImpl


def test_smoke():
    members_count = 3
    epochs = 5
    # shared uids also identifies dataset size
    shared_uids_count = 100
    batch_size = 10
    model_update_dim_size = 5

    shared_party_info = dict()

    master = MockPartyMasterImpl(
        uid="master",
        epochs=epochs,
        report_train_metrics_iteration=5,
        report_test_metrics_iteration=5,
        target=torch.rand(shared_uids_count),
        batch_size=batch_size,
        model_update_dim_size=model_update_dim_size
    )

    members = [
        MockPartyMemberImpl(
            uid=f"member-{i}",
            model_update_dim_size=model_update_dim_size
        )
        for i in range(members_count)
    ]

    def local_master_main():
        comm = LocalMasterPartyCommunicator(
            participant=master,
            world_size=members_count,
            shared_party_info=shared_party_info
        )
        comm.run()

    def local_member_main(member: PartyMember):
        comm = LocalMemberPartyCommunicator(
            participant=member,
            world_size=members_count,
            shared_party_info=shared_party_info
        )
        comm.run()

    threads = [
        Thread(name=f"main_{master.id}", target=local_master_main),
        *(
            Thread(
                name=f"main_{member.id}",
                target=local_member_main,
                args=(member,)
            )
            for member in members
        )
    ]

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()

    assert master.iteration_counter == epochs * math.ceil(shared_uids_count / batch_size)
    assert all([member.iterations_counter == epochs * math.ceil(shared_uids_count / batch_size) for member in members])
    assert master.is_initialized and master.is_finalized
    assert all([member.is_initialized and member.is_finalized for member in members])
