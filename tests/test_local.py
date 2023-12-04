from threading import Thread

import torch

from stalactite.communications.local import LocalMasterPartyCommunicator, LocalMemberPartyCommunicator
from stalactite.mocks import MockPartyMasterImpl, MockPartyMemberImpl


def test_smoke():
    members_count = 3

    shared_party_info = dict()

    def local_master_main(uid: str):
        comm = LocalMasterPartyCommunicator(
            participant=MockPartyMasterImpl(
                uid=uid,
                epochs=1,
                report_train_metrics_iteration=5,
                report_test_metrics_iteration=5,
                target=torch.randint(0, 2, (5,))
            ),
            world_size=members_count,
            shared_party_info=shared_party_info
        )
        comm.run()

    def local_member_main(member_id: str):
        comm = LocalMemberPartyCommunicator(
            participant=MockPartyMemberImpl(uid=member_id),
            world_size=members_count,
            shared_party_info=shared_party_info
        )
        comm.run()

    threads = [
        Thread(name="master_main", target=local_master_main, args=("master", members_count, shared_party_info)),
        *(
            Thread(
                name=f"member_main_{i}",
                target=local_member_main,
                args=(f"member-{i}", members_count, shared_party_info)
            )
            for i in range(members_count)
        )
    ]

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()

    # todo: add asserts
