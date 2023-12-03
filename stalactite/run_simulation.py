from threading import Thread

import torch

from party_master_Impl import PartyMasterImpl
from party_member_Impl import PartyMemberImpl
from party_Impl import PartyImpl
from stalactite.communications import LocalThreadBasedPartyCommunicator


def master_main():
    master = PartyMasterImpl(
        epochs=1,
        report_train_metrics_iteration=5,
        report_test_metrics_iteration=5,
        Y=torch.randint(0, 2, (5,))
    )
    comm = LocalThreadBasedPartyCommunicator()
    comm.run()


def member_main():
    member = PartyMemberImpl()
    comm = LocalThreadBasedPartyCommunicator()
    comm.run()


def main():
    member_count = 3

    threads = [
        Thread(name="master_main", target=master_main),
        *(Thread(name=f"member_main_{i}", target=member_main) for i in range(member_count))

    ]

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()


if __name__ == "__main__":
    main()
