import uuid
import random

import click
import torch

from stalactite.communications.distributed_grpc import GRpcMasterPartyCommunicator, GRpcMemberPartyCommunicator
from stalactite.mocks import MockPartyMasterImpl, MockPartyMemberImpl


members_count = 2
epochs = 5
# shared uids also identifies dataset size
# 1. target should be supplied with record uids
# 2. each member should have their own data (of different sizes)
# 3. uids mathching should be performed with relation to uids available for targets on master
# 4. target is also subject for filtering on master
# 5. shared_uids should identify uids that shared by all participants (including master itself)
# and should be generated beforehand
shared_uids_count = 100
batch_size = 10
model_update_dim_size = 5
# master + all members
num_target_records = 1000

num_dataset_records = [200 + random.randint(100, 1000) for _ in range(members_count)]
shared_record_uids = [str(i) for i in range(shared_uids_count)]
target_uids = [
    *shared_record_uids,
    *(str(uuid.uuid4()) for _ in range(num_target_records - len(shared_record_uids)))
]
members_datasets_uids = [
    [*shared_record_uids, *(str(uuid.uuid4()) for _ in range(num_records - len(shared_record_uids)))]
    for num_records in num_dataset_records
]
def grpc_master_main(uid: str, world_size: int):
    comm = GRpcMasterPartyCommunicator(
        participant=MockPartyMasterImpl(
            uid="master",
            epochs=epochs,
            report_train_metrics_iteration=5,
            report_test_metrics_iteration=5,
            target=torch.rand(shared_uids_count),
            target_uids=target_uids,
            batch_size=batch_size,
            model_update_dim_size=model_update_dim_size
        ),
        world_size=world_size,
        port='50051',
        host='0.0.0.0',
    )
    comm.run()


def grpc_member_main(member_id: int):
    comm = GRpcMemberPartyCommunicator(
        participant=MockPartyMemberImpl(
            uid=str(uuid.uuid4()),
            model_update_dim_size=model_update_dim_size,
            member_record_uids=members_datasets_uids[member_id]
        ),
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
@click.option('--member-id', type=int)
def run(member_id: int):
    grpc_member_main(member_id=member_id)


if __name__ == "__main__":
    cli()
