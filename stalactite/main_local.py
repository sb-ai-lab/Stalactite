from threading import Thread
from typing import Dict, Any

import torch

from party_master_Impl import PartyMasterImpl
from party_member_Impl import PartyMemberImpl
from stalactite.communications import LocalMasterPartyCommunicator, \
    LocalMemberPartyCommunicator


def master_main(world_size: int, shared_party_info: Dict[str, Any]):
    comm = LocalMasterPartyCommunicator(
        participant=PartyMasterImpl(
            epochs=1,
            report_train_metrics_iteration=5,
            report_test_metrics_iteration=5,
            Y=torch.randint(0, 2, (5,))
        ),
        world_size=world_size,
        shared_party_info=shared_party_info
    )
    comm.run()


def member_main(world_size: int, shared_party_info: Dict[str, Any]):
    comm = LocalMemberPartyCommunicator(
        participant=PartyMemberImpl(),
        world_size=world_size,
        shared_party_info=shared_party_info
    )
    comm.run()


def main():
    member_count = 3
    shared_party_info = dict()

    threads = [
        Thread(name="master_main", target=master_main, args=(member_count, shared_party_info)),
        *(
            Thread(name=f"member_main_{i}", target=member_main, args=(member_count, shared_party_info))
            for i in range(member_count)
        )
    ]

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()


if __name__ == "__main__":
    main()
