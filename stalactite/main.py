from threading import Thread
from typing import Dict, Any

import click
import torch

from stalactite.communications.local import LocalMasterPartyCommunicator, LocalMemberPartyCommunicator
from stalactite.mocks import PartyMasterImpl, MockPartyMemberImpl


def local_master_main(world_size: int, shared_party_info: Dict[str, Any]):
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


def local_member_main(member_id: str, world_size: int, shared_party_info: Dict[str, Any]):
    comm = LocalMemberPartyCommunicator(
        participant=MockPartyMemberImpl(uid=member_id),
        world_size=world_size,
        shared_party_info=shared_party_info
    )
    comm.run()


@click.group()
def cli():
    pass


@cli.group()
def local():
    pass


@local.command()
@click.option('--members-count', type=int, default=3)
def run(members_count: int):
    shared_party_info = dict()

    threads = [
        Thread(name="master_main", target=local_master_main, args=(members_count, shared_party_info)),
        *(
            Thread(
                name=f"member_main_{i}",
                target=local_member_main,
                args=(f"member_{i}", members_count, shared_party_info)
            )
            for i in range(members_count)
        )
    ]

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()


if __name__ == "__main__":
    cli()
