import logging
import os
from pathlib import Path
import subprocess
from threading import Thread
from typing import Dict, Any

import click
from docker import APIClient
from docker.errors import APIError
import torch

from stalactite.communications.local import LocalMasterPartyCommunicator, LocalMemberPartyCommunicator
from stalactite.mocks import MockPartyMasterImpl, MockPartyMemberImpl
from stalactite.utils import VFLConfig, load_yaml_config, raise_path_not_exist

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
sh = logging.StreamHandler()
sh.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(sh)

CONTAINER_LABEL = 'grpc-experiment'
TEST_CONTAINER_LABEL = 'test-grpc-experiment'


@click.group()
def cli():
    click.echo("Stalactite module API")


@cli.group()
def prerequisites():
    click.echo("Manage experimental prerequisites")


@prerequisites.command()
@click.option('--config-path', type=str)
@click.option(
    '-d',
    '--detached',
    is_flag=True,
    show_default=True,
    default=False,
    help="Run non-blocking start."
)
def start(config_path, detached):
    logger.info('Starting prerequisites containers')
    config = VFLConfig.model_validate(load_yaml_config(config_path))
    env_vars = os.environ.copy()
    env_vars['MLFLOW_PORT'] = config.prerequisites.mlflow_port
    command = f"{config.docker.docker_compose_command}"
    build_process = subprocess.run(
        command + ' build',
        cwd=config.docker.docker_compose_path,
        env=env_vars,
        shell=True,
    )
    try:
        build_process.check_returncode()
    except subprocess.CalledProcessError as exc:
        logger.error("Failed build process", exc_info=exc)
        raise
    up_process = subprocess.run(
        command + ' up' + (' -d' if detached else ''),
        cwd=config.docker.docker_compose_path,
        env=env_vars,
        shell=True,
    )
    try:
        up_process.check_returncode()
    except subprocess.CalledProcessError as exc:
        logger.error("Failed launch", exc_info=exc)
        raise

    if detached:
        logger.info(f"MlFlow port: {config.prerequisites.mlflow_port}")


@prerequisites.command()
@click.option('--config-path', type=str)
@click.option(
    '--remove',
    is_flag=True,
    show_default=True,
    default=False,
    help="Delete created containers, networks"
)
@click.option(
    '--remove-volumes',
    is_flag=True,
    show_default=True,
    default=False,
    help="Delete created volumes."
)
def stop(config_path, remove, remove_volumes):
    logger.info('Stopping prerequisites containers')
    config = VFLConfig.model_validate(load_yaml_config(config_path))
    env_vars = os.environ.copy()
    env_vars['MLFLOW_PORT'] = config.prerequisites.mlflow_port
    # TODO prometheus
    command = f"{config.docker.docker_compose_command}"
    build_process = subprocess.run(
        command + ' stop',
        cwd=config.docker.docker_compose_path,
        env=env_vars,
        shell=True,
    )
    try:
        build_process.check_returncode()
    except subprocess.CalledProcessError as exc:
        logger.error("Failed stopping prerequisites containers", exc_info=exc)
        raise
    if remove:
        down_process = subprocess.run(
            command + ' down' + (' -v' if remove_volumes else ''),
            cwd=config.docker.docker_compose_path,
            env=env_vars,
            shell=True,
        )
        try:
            down_process.check_returncode()
        except subprocess.CalledProcessError as exc:
            logger.error("Failed releasing resources", exc_info=exc)
            raise

        logger.info(f"Successfully teared down prerequisite resources")


@cli.group()
def member():
    click.echo("Manage distributed (multi-node) members")


@member.command()
@click.option('--config-path', type=str)
def start(config_path):
    # TODO
    raise NotImplementedError


@member.command()
@click.option('--config-path', type=str)
def stop(config_path):
    # TODO
    raise NotImplementedError


@member.command()
@click.option('--config-path', type=str)
def status(config_path):
    # TODO
    raise NotImplementedError


@member.command()
@click.option('--config-path', type=str)
def logs(config_path):
    # TODO
    raise NotImplementedError


@cli.group()
def master():
    click.echo("Manage distributed (multi-node) master")


@master.command()
@click.option('--config-path', type=str)
def start(config_path):
    # TODO
    raise NotImplementedError


@master.command()
@click.option('--config-path', type=str)
def stop(config_path):
    # TODO
    raise NotImplementedError


@master.command()
@click.option('--config-path', type=str)
def status(config_path):
    # TODO
    raise NotImplementedError


@master.command()
@click.option('--config-path', type=str)
def logs(config_path):
    # TODO
    raise NotImplementedError


@cli.group()
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
@click.pass_context
def local(ctx, single_process, multi_process):
    if multi_process and not single_process:
        click.echo('Multiple-process single-node mode')
    elif single_process and not multi_process:
        click.echo('Multiple-threads single-node mode')
    else:
        raise SyntaxError('Either `--single-process` or `--multi-process` flag can be set.')
    ctx.obj = dict()
    ctx.obj['multi_process'] = multi_process
    ctx.obj['single_process'] = single_process


@local.command()
@click.option('--config-path', type=str)
@click.pass_context
def start(ctx, config_path):
    # TODO
    raise NotImplementedError


@local.command()
@click.option('--config-path', type=str)
def stop(config_path):
    # TODO
    raise NotImplementedError


@local.command()
@click.option('--config-path', type=str)
def status(config_path):
    # TODO
    raise NotImplementedError


@local.command()
@click.option('--config-path', type=str)
def logs(config_path):
    # TODO
    raise NotImplementedError


@cli.group()
@click.pass_context
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
def test(ctx, multi_process, single_process):
    if multi_process and not single_process:
        click.echo('Manage multi-process single-node tests')
    elif single_process and not multi_process:
        click.echo('Manage single-process (multi-thread) single-node tests')
    else:
        raise SyntaxError('Either `--single-process` or `--multi-process` flag can be set.')
    ctx.obj = dict()
    ctx.obj['multi_process'] = multi_process
    ctx.obj['single_process'] = single_process


@test.command()
@click.pass_context
@click.option('--config-path', type=str)
def start(ctx, config_path):
    config = VFLConfig.model_validate(load_yaml_config(config_path))
    if ctx.obj['multi_process'] and not ctx.obj['single_process']:
        logger.info('Starting multi-process single-node test')
        client = APIClient(base_url='unix://var/run/docker.sock')
        image_name = 'grpc-base.dockerfile'
        image = 'grpc-base:latest'
        try:
            logger.info('Building image of the agent')
            logs = client.build(
                path=str(Path(os.path.abspath(__file__)).parent.parent),
                tag=image,
                quiet=True,
                decode=True,
                nocache=False,
                dockerfile=os.path.join(Path(os.path.abspath(__file__)).parent.parent, 'docker', image_name),
            )
            for log in logs:
                logger.debug(log["stream"])
        except APIError as exc:
            logger.error('Error while building an image', exc_info=exc)
            raise
        network = 'vfl-network'
        if networks := client.networks(names=[network]):
            network = networks.pop()['Name']
        else:
            client.create_network(network)
        networking_config = client.create_networking_config({
            network: client.create_endpoint_config()
        })

        data_path = config.data.host_path_data_dir
        configs_path = os.path.dirname(os.path.abspath(config_path))

        raise_path_not_exist(configs_path)
        raise_path_not_exist(data_path)

        volumes = [f'{data_path}', f'{configs_path}']
        mounts_host_config = client.create_host_config(binds=[
            f'{configs_path}:{configs_path}:rw',
            f'{data_path}:{data_path}:rw',
        ])
        try:
            logger.info('Starting gRPC master container')
            grpc_server_host = 'grpc-master'
            master_container = client.create_container(
                image=image,
                command=["python", "run_grpc_master.py", "--config-path", f"{os.path.abspath(config_path)}"],
                detach=True,
                environment={},
                hostname=grpc_server_host,
                labels={'container-g': TEST_CONTAINER_LABEL},
                volumes=volumes,
                host_config=mounts_host_config,
                networking_config=networking_config,
            )
            logger.info(f'Master container id: {master_container.get("Id")}')
            client.start(container=master_container.get('Id'))

            for member_rank in range(config.common.world_size):
                logger.info(f'Starting gRPC member-{member_rank} container')

                member_container = client.create_container(
                    image=image,
                    command=["python", "run_grpc_member.py", "--config-path", f"{os.path.abspath(config_path)}"],
                    detach=True,
                    environment={'RANK': member_rank, 'GRPC_SERVER_HOST': grpc_server_host},
                    labels={'container-g': TEST_CONTAINER_LABEL},
                    volumes=volumes,
                    host_config=mounts_host_config,
                    networking_config=networking_config,
                )
                logger.info(f'Member {member_rank} container id: {member_container.get("Id")}')
                client.start(container=member_container.get('Id'))
        except APIError as exc:
            logger.error('Error while agents containers launch', exc_info=exc)
            raise
    elif ctx.obj['single_process'] and not ctx.obj['multi_process']:
        # TODO bufix and refactor
        def local_master_main(uid: str, world_size: int, shared_party_info: Dict[str, Any]):
            comm = LocalMasterPartyCommunicator(
                participant=MockPartyMasterImpl(
                    uid=uid,
                    epochs=1,
                    report_train_metrics_iteration=5,
                    report_test_metrics_iteration=5,
                    target=torch.randint(0, 2, (5,))
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

        logger.info('Starting multi-thread single-process test')
        shared_party_info = dict()
        members_count = config.common.world_size
        threads = [
            Thread(
                name="master_main",
                daemon=True,
                target=local_master_main,
                args=("master", members_count, shared_party_info)
            ),
            *(
                Thread(
                    name=f"member_main_{i}",
                    daemon=True,
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


@test.command()
@click.pass_context
@click.option(
    '--leave-containers',
    is_flag=True,
    show_default=False,
    default=False,
    help="Delete created agents containers"
)
def stop(ctx, leave_containers):
    if ctx.obj['multi_process'] and not ctx.obj['single_process']:
        logger.info('Stopping multi-process single-node test')
        client = APIClient(base_url='unix://var/run/docker.sock')
        try:
            containers = client.containers(all=True, filters={'label': f'container-g={TEST_CONTAINER_LABEL}'})
            for container in containers:
                logger.info(f'Stopping {container["Id"]}')
                client.stop(container)
                if not leave_containers:
                    logger.info(f'Removing {container["Id"]}')
                    client.remove_container(container)
        except APIError as exc:
            logger.error('Error while stopping and removing containers', exc_info=exc)
            raise


@test.command()
@click.option('--agent-id', type=str, default=None)
def status(agent_id):
    client = APIClient(base_url='unix://var/run/docker.sock')
    try:
        if agent_id is None:
            containers = client.containers(all=True, filters={'label': f'container-g={TEST_CONTAINER_LABEL}'})
        else:
            containers = client.containers(all=True, filters={'id': agent_id})
        for container in containers:
            logger.info(f'Container id: {container["Id"]}. Status: {container["Status"]}')
    except APIError as exc:
        logger.error('Error checking container status', exc_info=exc)
        raise


@test.command()
@click.option('--agent-id', type=str)
@click.option(
    '--follow',
    is_flag=True,
    show_default=True,
    default=False,
    help="Delete created agents containers"
)
@click.option('--tail', type=str, default='all')
def logs(agent_id, follow, tail):
    client = APIClient(base_url='unix://var/run/docker.sock')
    if tail != 'all':
        try:
            tail = int(tail)
        except ValueError:
            logger.warning('Invalid `tail` argument. Must be positive int or `all`. Defaulting to `10`.')
            tail = 10
    try:
        logs_gen = client.logs(container=agent_id, follow=follow, stream=True, tail=tail)
        for log in logs_gen:
            print(log.decode().strip())
    except APIError as exc:
        logger.error('Error retrieving container logs', exc_info=exc)
        raise


@cli.command()
@click.option('--config-path', type=str)
def predict():
    click.echo('Run VFL predictions')
    # TODO
    raise NotImplementedError


@cli.group()
def report():
    click.echo('Get report on the experiment')


@report.command()
@click.option('--config-path', type=str)
@click.option('--report-save-path', type=str)
def export():
    # TODO
    raise NotImplementedError


@report.command()
@click.option('--config-path', type=str)
def mlflow():
    # TODO
    raise NotImplementedError


@report.command()
@click.option('--config-path', type=str)
def prometheus():
    # TODO
    raise NotImplementedError


if __name__ == "__main__":
    cli()
