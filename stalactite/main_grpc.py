import uuid
import os
import random
import logging
from pathlib import Path

import click
import torch
import docker
from docker import APIClient
from docker.types import LogConfig

from stalactite.communications.distributed_grpc import GRpcMasterPartyCommunicator, GRpcMemberPartyCommunicator
from stalactite.mocks import MockPartyMasterImpl, MockPartyMemberImpl
from stalactite.utils import VFLConfig, load_yaml_config

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
sh = logging.StreamHandler()
sh.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(sh)

members_count = 2
epochs = 2
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


@cli.group()
def test():
    click.echo("Run tests of stalactite")


@test.group()
def local():
    click.echo("Local test mode")


@test.group()
def distributed():
    click.echo("Distributed test mode")


@cli.group()
def local():
    click.echo("Local (single-node) experiment run.")


def _raise_path_not_exist(path: str):
    if not os.path.exists(path):
        raise FileExistsError(f'Path {path} does not exist')


@test.command()
@click.option(
    '--single-process',
    is_flag=True,
    show_default=True,
    default=False,
    help="Run single-node single-process (multi-thread) test."
)
@click.option(
    '--multi-process',
    is_flag=True,
    show_default=True,
    default=False,
    help="Run single-node multi-process (dockerized) test."
)
@click.option('--config-path', type=str, default='../configs/config.yml')
def run(config_path, single_process, multi_process):
    if multi_process and not single_process:
        logger.info('Running multi process test.')

        client = APIClient(base_url='unix://var/run/docker.sock')

        config = VFLConfig.model_validate(load_yaml_config(config_path))
        image_name = 'grpc-base.dockerfile'
        image = 'grpc-base:latest'
        logger.info('Building image of the agent')

        logs = client.build(
            path=str(Path(os.path.abspath(__file__)).parent.parent),
            tag=image,
            quiet=True,
            timeout=600,
            decode=True,
            nocache=False,
            dockerfile=os.path.join(Path(os.path.abspath(__file__)).parent.parent, 'docker', image_name),
        )
        for log in logs:
            logger.debug(log["stream"])
        network = 'vfl-network'
        if networks := client.networks(names=[network]):
            network = networks.pop()['Name']
        else:
            client.create_network(network)
        networking_config = client.create_networking_config({
            network: client.create_endpoint_config()
        })

        configs_path = '/mnt/ess_storage/DN_1/storage/qa-system-research/zakharova/vfl-benchmark-test-mounts/configs'
        data_path = '/mnt/ess_storage/DN_1/storage/qa-system-research/zakharova/vfl-benchmark-test-mounts/data' # TODO into config
        # TODO change

        # _raise_path_not_exist(configs_path)
        # _raise_path_not_exist(data_path)

        volumes = [f'/mnt/configs', f'/mnt/data']
        mounts_host_config = client.create_host_config(binds=[
            f'{configs_path}:/mnt/configs:rw',
            f'{data_path}:/mnt/data:rw',
        ])

        logger.info('Starting gRPC master container')
        grpc_server_host = 'grpc-master'
        master_container = client.create_container(
            image=image,
            command=["python", "run_grpc_master.py", "--config-path", "/mnt/configs/config-test.yml"],
            detach=True,
            environment={},
            hostname=grpc_server_host,
            labels={'container-g': 'grpc-experiment'},
            volumes=[f'/mnt/configs', f'/mnt/data'],
            host_config=mounts_host_config,
            networking_config=networking_config,
        )
        logger.info(f'Master container id: {master_container.get("Id")}')
        client.start(container=master_container.get('Id'))

        for member_rank in range(config.common.world_size):
            logger.info(f'Starting gRPC member-{member_rank} container')

            member_container = client.create_container(
                image=image,
                command=["python", "run_grpc_member.py", "--config-path", "/mnt/configs/config-test.yml"],
                detach=True,
                environment={'RANK': member_rank, 'GRPC_SERVER_HOST': grpc_server_host},
                labels={'container-g': 'grpc-experiment'},
                volumes=volumes,
                host_config=mounts_host_config,
                networking_config=networking_config,
            )
            logger.info(f'Member {member_rank} container id: {member_container.get("Id")}')
            client.start(container=member_container.get('Id'))

    elif single_process and not multi_process:
        raise NotImplementedError
    else:
        raise SyntaxError('Either `--single-process` or `--multi-process` flag can be set.')


if __name__ == "__main__":
    cli()
