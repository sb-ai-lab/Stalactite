import uuid

import click
import torch

from stalactite.communications.distributed_grpc import GRpcMasterPartyCommunicator, GRpcMemberPartyCommunicator
from stalactite.mocks import MockPartyMasterImpl, MockPartyMemberImpl


def grpc_master_main(uid: str, world_size: int):
    comm = GRpcMasterPartyCommunicator(
        participant=MockPartyMasterImpl(
            uid=uid,
            epochs=1,
            report_train_metrics_iteration=5,
            report_test_metrics_iteration=5,
            target=torch.randint(0, 2, (5,))
        ),
        world_size=world_size,
        port='50051',
        host='0.0.0.0',
    )
    comm.run()


def grpc_member_main():
    comm = GRpcMemberPartyCommunicator(
        participant=MockPartyMemberImpl(uid=str(uuid.uuid4())),
        master_host='0.0.0.0',
        master_port='50051',
    )
    comm.run()


@click.group()
def cli():
    pass


@cli.group()
def master():
    click.echo("Run distributed process master")


@master.command()
@click.option('--members-count', type=int, default=3)
def run(members_count: int):
    grpc_master_main("master", members_count)


@cli.group()
def member():
    click.echo("Run distributed process member")


@member.command()
def run():
    grpc_member_main()


if __name__ == "__main__":
    cli()
