import logging
import os
from pathlib import Path
from threading import Thread
from typing import Dict, Any

import click
from docker import APIClient
from docker.errors import APIError
import mlflow as _mlflow
import torch

from stalactite.communications.local import LocalMasterPartyCommunicator, LocalMemberPartyCommunicator
from stalactite.mocks import MockPartyMasterImpl, MockPartyMemberImpl
from stalactite.configs import VFLConfig, raise_path_not_exist
from stalactite.utils_main import (
    get_env_vars,
    run_subprocess_command,
    is_test_environment,
    build_base_image,
    get_logs,
    get_status,
    stop_containers,
    BASE_CONTAINER_LABEL,
    KEY_CONTAINER_LABEL,
    BASE_MASTER_CONTAINER_NAME,
    BASE_MEMBER_CONTAINER_NAME,
    BASE_IMAGE_FILE,
    BASE_IMAGE_TAG,
    PREREQUISITES_NETWORK,

)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
sh = logging.StreamHandler()
sh.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(sh)


@click.group()
def cli():
    click.echo("Stalactite module API")


@cli.group()
def prerequisites():
    """ Prerequisites management command group. """
    click.echo("Manage experimental prerequisites")


@prerequisites.command()
@click.option('--config-path', type=str, required=True)
@click.option(
    '-d',
    '--detached',
    is_flag=True,
    show_default=True,
    default=False,
    help="Run non-blocking start."
)
def start(config_path, detached):
    """
    Start containers with prerequisites (MlFlow, Prometheus, Grafana).

    :param config_path: Path to the VFL config file
    :param detached: Start containers in detached regime
    """
    logger.info('Starting prerequisites containers')
    config = VFLConfig.load_and_validate(config_path)
    env_vars = get_env_vars(config)
    command = f"{config.docker.docker_compose_command}"
    run_subprocess_command(
        command=command + ' up' + (' -d' if detached else '') + ' --build',
        logger_err_info="Failed build process",
        cwd=config.docker.docker_compose_path,
        env=env_vars,
        shell=True
    )
    if detached:
        logger.info(
            f"MlFlow port: {config.prerequisites.mlflow_port}\n"
            f"Prometheus port: {config.prerequisites.prometheus_port}\n"
            f"Grafana port: {config.prerequisites.grafana_port}"
        )


@prerequisites.command()
@click.option('--config-path', type=str, required=True)
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
    """
    Stop prerequisites containers (and remove all defined in docker-compose.yml networks and images[, and volumes]).

    :param config_path: Path to the VFL config file
    :param remove: Remove created containers, networks
    :param remove_volumes: Remove volumes
    """
    logger.info('Stopping prerequisites containers')
    config = VFLConfig.load_and_validate(config_path)
    env_vars = get_env_vars(config)
    command = f"{config.docker.docker_compose_command}"
    run_subprocess_command(
        command=command + ' stop',
        logger_err_info="Failed stopping prerequisites containers",
        cwd=config.docker.docker_compose_path,
        env=env_vars,
        shell=True
    )
    logger.info(f"Successfully stopped prerequisite containers")
    if remove:
        run_subprocess_command(
            command=command + ' down' + (' -v' if remove_volumes else ''),
            logger_err_info="Failed releasing resources",
            cwd=config.docker.docker_compose_path,
            env=env_vars,
            shell=True
        )
        logger.info(f"Successfully teared down prerequisite resources")


@cli.group()
@click.pass_context
def member(ctx):
    click.echo("Manage distributed (multi-node) members")
    ctx.obj = dict()
    postfix = '-distributed'
    ctx.obj['member_container_label'] = BASE_CONTAINER_LABEL + postfix + '-member'
    ctx.obj['member_container_name'] = lambda rank: BASE_MEMBER_CONTAINER_NAME + f'-{rank}' + postfix


@member.command()
@click.option('--config-path', type=str)
@click.option(
    '--rank',
    type=int,
    help='Rank of the member for correct data loading'
)  # TODO remove after refactoring of the preprocessing
@click.option(
    '-d',
    '--detached',
    is_flag=True,
    show_default=True,
    default=False,
    help="Run non-blocking start."
)
@click.pass_context
def start(ctx, config_path, rank, detached):
    config = VFLConfig.load_and_validate(config_path)
    logger.info('Starting member for distributed experiment')
    client = APIClient(base_url='unix://var/run/docker.sock')
    logger.info('Building image for the member container. If build for the first time, it may take a while...')
    try:
        build_base_image(client, logger=logger)

        logger.info(f'Starting gRPC member-{rank} container')

        data_path = config.data.host_path_data_dir
        configs_path = os.path.dirname(os.path.abspath(config_path))
        raise_path_not_exist(configs_path)
        raise_path_not_exist(data_path)

        member_container = client.create_container(
            image=BASE_IMAGE_TAG,
            command=["python", "run_grpc_member.py", "--config-path", f"{os.path.abspath(config_path)}"],
            detach=True,
            environment={'RANK': rank},
            labels={KEY_CONTAINER_LABEL: ctx.obj['member_container_label']},
            volumes=[f'{data_path}', f'{configs_path}'],
            host_config=client.create_host_config(
                binds=[
                    f'{configs_path}:{configs_path}:rw',
                    f'{data_path}:{data_path}:rw',
                ],
            ),
            name=ctx.obj['member_container_name'](rank),
        )
        logger.info(f'Member {rank} container id: {member_container.get("Id")}')
        client.start(container=member_container.get('Id'))
        if not detached:
            output = client.attach(member_container, stream=True, logs=True)
            try:
                for log in output:
                    print(log.decode().strip())
            except KeyboardInterrupt:
                client.stop(member_container)  # TODO question: stop if interrupted?
    except APIError as exc:
        logger.error(f'Error while starting member-{rank} container', exc_info=exc)
        raise


@member.command()
@click.option(
    '--leave-containers',
    is_flag=True,
    show_default=False,
    default=False,
    help="Flag to retain created agents containers"
)
@click.pass_context
def stop(ctx, leave_containers):
    logger.info('Stopping members` containers')
    client = APIClient(base_url='unix://var/run/docker.sock')
    try:
        container_label = ctx.obj['member_container_label']
        containers = client.containers(all=True, filters={'label': f'{KEY_CONTAINER_LABEL}={container_label}'})
        stop_containers(client, containers, leave_containers=leave_containers, logger=logger)
    except APIError as exc:
        logger.error('Error while stopping (and removing) containers', exc_info=exc)


@member.command()
@click.option('--agent-id', type=str, default=None)
@click.option('--rank', type=str, default=None)
@click.pass_context
def status(ctx, agent_id, rank):
    client = ctx.obj.get('client', APIClient(base_url='unix://var/run/docker.sock'))
    container_label = ctx.obj['member_container_label']
    if agent_id is None and rank is None:
        get_status(
            agent_id=None,
            logger=logger,
            containers_label=f'{KEY_CONTAINER_LABEL}={container_label}',
            docker_client=client
        )
        return
    elif agent_id is not None:
        container_name_or_id = agent_id
    else:
        container_name_or_id = ctx.obj['member_container_name'](rank)
    get_status(
        agent_id=container_name_or_id,
        logger=logger,
        containers_label=f'{KEY_CONTAINER_LABEL}={container_label}',
        docker_client=client
    )


@member.command()
@click.option('--agent-id', type=str, default=None)
@click.option('--rank', type=str, default=None)  # TODO rm after refactor?
@click.option(
    '--follow',
    is_flag=True,
    show_default=True,
    default=False,
    help="Follow log output"
)
@click.option('--tail', type=str, default='all')
@click.pass_context
def logs(ctx, agent_id, rank, follow, tail):
    if agent_id is None and rank is None:
        raise SyntaxError('Either `--agent-id` or `--rank` must be passed.')
    if agent_id is not None:
        container_name_or_id = agent_id
    else:
        container_name_or_id = ctx.obj['member_container_name'](rank)
    client = APIClient(base_url='unix://var/run/docker.sock')
    get_logs(
        agent_id=container_name_or_id,
        tail=tail,
        follow=follow,
        docker_client=client,
    )


@cli.group()
@click.pass_context
def master(ctx):
    click.echo("Manage distributed (multi-node) master")
    ctx.obj = dict()
    postfix = '-distributed'
    ctx.obj['master_container_label'] = BASE_CONTAINER_LABEL + postfix + '-master'
    ctx.obj['master_container_name'] = BASE_MASTER_CONTAINER_NAME + postfix


@master.command()
@click.option('--config-path', type=str)
@click.option(
    '-d',
    '--detached',
    is_flag=True,
    show_default=True,
    default=False,
    help="Run non-blocking start."
)
@click.pass_context
def start(ctx, config_path, detached):
    config = VFLConfig.load_and_validate(config_path)
    logger.info('Starting master for distributed experiment')
    client = APIClient(base_url='unix://var/run/docker.sock')
    logger.info('Building image for the master container. If build for the first time, it may take a while...')
    build_base_image(client, logger=logger)

    data_path = config.data.host_path_data_dir
    configs_path = os.path.dirname(os.path.abspath(config_path))
    raise_path_not_exist(configs_path)
    raise_path_not_exist(data_path)

    try:
        logger.info('Starting gRPC master container')

        master_container = client.create_container(
            image=BASE_IMAGE_TAG,
            command=["python", "run_grpc_master.py", "--config-path", f"{os.path.abspath(config_path)}"],
            detach=True,
            labels={KEY_CONTAINER_LABEL: ctx.obj['master_container_label']},
            volumes=[f'{data_path}', f'{configs_path}'],
            host_config=client.create_host_config(
                binds=[
                    f'{configs_path}:{configs_path}:rw',
                    f'{data_path}:{data_path}:rw',
                ],
                port_bindings={config.grpc_server.port: config.grpc_server.port},
            ),
            ports=[config.grpc_server.port],
            name=ctx.obj['master_container_name'],
        )
        client.start(container=master_container.get('Id'))
        logger.info(f'Master container id: {master_container.get("Id")}')
        if not detached:
            output = client.attach(master_container, stream=True, logs=True)
            try:
                for log in output:
                    print(log.decode().strip())
            except KeyboardInterrupt:
                client.stop(master_container)  # TODO question: stop if interrupted?

    except APIError as exc:
        logger.error('Error while starting master container', exc_info=exc)
        raise


@master.command()
@click.option(
    '--leave-containers',
    is_flag=True,
    show_default=False,
    default=False,
    help="Flag to retain created agents containers"
)
@click.pass_context
def stop(ctx, leave_containers):
    logger.info('Stopping master container')
    client = ctx.obj.get('client', APIClient(base_url='unix://var/run/docker.sock'))
    try:
        containers = client.containers(all=True, filters={'name': ctx.obj['master_container_name']})
        if len(containers) < 1:
            logger.warning('Found 0 containers. Skipping.')
            return
        stop_containers(client, containers, leave_containers=leave_containers, logger=logger)
    except APIError as exc:
        logger.error('Error while stopping (and removing) master container', exc_info=exc)


@master.command()
@click.pass_context
def status(ctx):
    container_label = ctx.obj['master_container_label']
    client = ctx.obj.get('client', APIClient(base_url='unix://var/run/docker.sock'))
    get_status(
        agent_id=None,
        logger=logger,
        containers_label=f'{KEY_CONTAINER_LABEL}={container_label}',
        docker_client=client
    )


@master.command()
@click.option(
    '--follow',
    is_flag=True,
    show_default=True,
    default=False,
    help="Follow log output"
)
@click.option('--tail', type=str, default='all')
@click.pass_context
def logs(ctx, follow, tail):
    client = APIClient(base_url='unix://var/run/docker.sock')
    get_logs(
        agent_id=ctx.obj['master_container_name'],
        tail=tail,
        follow=follow,
        docker_client=client,
    )


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
    """
    Local experiments (multi-process / single process) mode command group.

    :param ctx: Click context passed through into group`s commands
    :param single_process: Run single process experiment
    :param multi_process: Run multiple process (dockerized) experiment
    """
    ctx.obj = dict()
    if multi_process and not single_process:
        click.echo('Multiple-process single-node mode')
        ctx.obj['client'] = APIClient(base_url='unix://var/run/docker.sock')
    elif single_process and not multi_process:
        click.echo('Multiple-threads single-node mode')
    else:
        raise SyntaxError('Either `--single-process` or `--multi-process` flag can be set.')
    ctx.obj['multi_process'] = multi_process
    ctx.obj['single_process'] = single_process


@local.command()
@click.option('--config-path', type=str, required=True)
@click.pass_context
def start(ctx, config_path):
    """
    Start local experiment.
    For a multiprocess mode build and start VFL master and members containers.

    :param ctx: Passed from group Click context
    :param config_path: Path to the VFL config file
    """
    _test = is_test_environment()
    config = VFLConfig.load_and_validate(config_path)
    if ctx.obj['multi_process'] and not ctx.obj['single_process']:
        logger.info('Starting multi-process single-node experiment')
        client = ctx.obj.get('client', APIClient(base_url='unix://var/run/docker.sock'))
        logger.info('Building image of the agent. If build for the first time, it may take a while...')
        build_base_image(client, logger=logger)

        if networks := client.networks(names=[PREREQUISITES_NETWORK]):
            network = networks.pop()['Name']
        else:
            network = PREREQUISITES_NETWORK
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

        container_label = BASE_CONTAINER_LABEL + ('-test' if _test else '')
        master_container_name = BASE_MASTER_CONTAINER_NAME + ('-test' if _test else '')
        try:
            logger.info('Starting gRPC master container')
            grpc_server_host = 'grpc-master'

            master_container = client.create_container(
                image=BASE_IMAGE_TAG,
                command=["python", "run_grpc_master.py", "--config-path", f"{os.path.abspath(config_path)}"],
                detach=True,
                environment={},
                hostname=grpc_server_host,
                labels={KEY_CONTAINER_LABEL: container_label},
                volumes=volumes,
                host_config=mounts_host_config,
                networking_config=networking_config,
                name=master_container_name,
            )
            logger.info(f'Master container id: {master_container.get("Id")}')
            client.start(container=master_container.get('Id'))

            for member_rank in range(config.common.world_size):
                logger.info(f'Starting gRPC member-{member_rank} container')
                member_container_name = BASE_MEMBER_CONTAINER_NAME + f'-{member_rank}' + ('-test' if _test else '')

                member_container = client.create_container(
                    image=BASE_IMAGE_TAG,
                    command=["python", "run_grpc_member.py", "--config-path", f"{os.path.abspath(config_path)}"],
                    detach=True,
                    environment={'RANK': member_rank, 'GRPC_SERVER_HOST': grpc_server_host},
                    labels={KEY_CONTAINER_LABEL: container_label},
                    volumes=volumes,
                    host_config=mounts_host_config,
                    networking_config=networking_config,
                    name=member_container_name,
                )
                logger.info(f'Member {member_rank} container id: {member_container.get("Id")}')
                client.start(container=member_container.get('Id'))
        except APIError as exc:
            logger.error(
                'Error while agents containers launch. If the container name is in use, alternatively to renaming you '
                'can remove all containers from previous experiment by running '
                f'`stalactite {"test" if _test else "local"} --multi-process stop`',
                exc_info=exc
            )
            raise
    elif ctx.obj['single_process'] and not ctx.obj['multi_process']:
        # TODO bufix and refactor
        raise NotImplementedError

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


@local.command()
@click.option(
    '--leave-containers',
    is_flag=True,
    show_default=False,
    default=False,
    help="Flag to retain created agents containers"
)
@click.pass_context
def stop(ctx, leave_containers):
    """
    Stop local experiment.
    For a multiprocess mode stop and remove containers of the VFL master and members.

    :param ctx: Passed from group Click context
    :param leave_containers: Whether to leave stopped containers without removing them
    """
    _test = is_test_environment()
    if ctx.obj['multi_process'] and not ctx.obj['single_process']:
        logger.info('Stopping multi-process single-node')
        client = ctx.obj.get('client', APIClient(base_url='unix://var/run/docker.sock'))
        try:
            container_label = BASE_CONTAINER_LABEL + ('-test' if _test else '')
            containers = client.containers(all=True, filters={'label': f'{KEY_CONTAINER_LABEL}={container_label}'})
            stop_containers(client, containers, leave_containers=leave_containers, logger=logger)
        except APIError as exc:
            logger.error('Error while stopping (and removing) containers', exc_info=exc)


@local.command()
@click.option('--agent-id', type=str, default=None)
@click.pass_context
def status(ctx, agent_id):
    """
    Print status of the experimental container(s).

    :param ctx: Passed from group Click context
    :param agent_id: ID of the agent to print a status for
    """
    _test = is_test_environment()
    container_label = BASE_CONTAINER_LABEL + ('-test' if _test else '')
    client = ctx.obj.get('client', APIClient(base_url='unix://var/run/docker.sock'))
    get_status(
        agent_id=agent_id,
        logger=logger,
        containers_label=f'{KEY_CONTAINER_LABEL}={container_label}',
        docker_client=client
    )


@local.command()
@click.option('--agent-id', type=str, required=True)
@click.option(
    '--follow',
    is_flag=True,
    show_default=True,
    default=False,
    help="Follow log output"
)
@click.option('--tail', type=str, default='all')
@click.pass_context
def logs(ctx, agent_id, follow, tail):
    """
    Print logs of the experimental container.

    :param ctx: Passed from group Click context
    :param agent_id: ID of the agent to print a status for
    :param follow: Whether to continue streaming the new output from the logs
    :param tail: Number of lines to show from the end of the logs
    """
    client = ctx.obj.get('client', APIClient(base_url='unix://var/run/docker.sock'))
    get_logs(
        agent_id=agent_id,
        tail=tail,
        follow=follow,
        docker_client=client,
    )


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
    """
    Local tests (multi-process / single process) mode command group.

    :param ctx: Click context passed through into group`s commands
    :param single_process: Run tests on single process experiment
    :param multi_process: Run tests on multiple process (dockerized) experiment
    """
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
@click.option('--config-path', type=str, required=True)
def start(ctx, config_path):
    """
    Start local test experiment.
    For a multiprocess mode run integration tests on started VFL master and members containers.

    :param ctx: Passed from group Click context
    :param config_path: Path to the VFL config file
    """
    config = VFLConfig.load_and_validate(config_path)
    if ctx.obj['multi_process'] and not ctx.obj['single_process']:
        test_group_name = 'TestLocalGroupStart'
        report_file_name = f'{test_group_name}-log-{config.common.experiment_label}.jsonl'
        run_subprocess_command(
            command=f"python -m pytest --test_config_path {config_path} "
                    f"tests/distributed_grpc/integration_test.py -k '{test_group_name}' -x "
                    f"--report-log={os.path.join(config.common.reports_export_folder, report_file_name)} "
                    "-W ignore::DeprecationWarning",
            logger_err_info="Failed running test",
            cwd=Path(__file__).parent.parent,
            shell=True
        )
    else:
        # TODO
        raise NotImplementedError


@test.command()
@click.pass_context
@click.option('--config-path', type=str)
@click.option(
    '--no-tests',
    is_flag=True,
    show_default=True,
    default=False,
    help="Remove test containers without launching Pytest."
)
def stop(ctx, config_path, no_tests):
    """
    Stop local test experiment.
    For a multiprocess mode run integration tests on stopped VFL master and members containers.

    :param ctx: Passed from group Click context
    :param config_path: Path to the VFL config file
    :param no_tests: Whether to discard testing and remove test containers only
    """
    if config_path is None and not no_tests:
        raise SyntaxError('Specify `--config-path` or pass flag `--no-tests`')
    if no_tests:
        _test = 1
        if ctx.obj['multi_process'] and not ctx.obj['single_process']:
            logger.info('Removing test containers')
            client = ctx.obj.get('client', APIClient(base_url='unix://var/run/docker.sock'))
            try:
                container_label = BASE_CONTAINER_LABEL + ('-test' if _test else '')
                containers = client.containers(all=True, filters={'label': f'{KEY_CONTAINER_LABEL}={container_label}'})
                stop_containers(client, containers, leave_containers=False, logger=logger)
            except APIError as exc:
                logger.error('Error while stopping (and removing) containers', exc_info=exc)
            return
    config = VFLConfig.load_and_validate(config_path)
    if ctx.obj['multi_process'] and not ctx.obj['single_process']:
        test_group_name = 'TestLocalGroupStop'
        report_file_name = f'{test_group_name}-log-{config.common.experiment_label}.jsonl'
        run_subprocess_command(
            command=f"python -m pytest --test_config_path {config_path} "
                    f"tests/distributed_grpc/integration_test.py -k '{test_group_name}' -x "
                    f"--report-log={os.path.join(config.common.reports_export_folder, report_file_name)} "
                    "-W ignore::DeprecationWarning",
            logger_err_info="Failed running test",
            cwd=Path(__file__).parent.parent,
            shell=True
        )


@test.command()
@click.option('--agent-id', type=str, default=None)
def status(agent_id):
    """
    Print status of the experimental test container(s).

    :param ctx: Passed from group Click context
    :param agent_id: ID of the agent to print a status for
    """
    _test = True
    container_label = BASE_CONTAINER_LABEL + ('-test' if _test else '')
    get_status(agent_id=agent_id, containers_label=f'{KEY_CONTAINER_LABEL}={container_label}', logger=logger)


@test.command()
@click.option(
    '--agent-id',
    type=str,
    default=None,
    help='If the `agent-id` is passed, show container logs, otherwise, prints test report')
@click.option('--tail', type=str, default='all')
@click.option('--config-path', type=str, default=None)
def logs(agent_id, config_path, tail):
    """
    Print logs of the experimental test container or return path to tests` logs.

    :param agent_id: ID of the agent to print a status for (for container logs). If not passed, print Pytest report
    :param config_path: Path to the VFL config file
    :param tail: Number of lines to show from the end of the logs (for container logs)
    """
    if agent_id is None and config_path is None:
        raise SyntaxError('Either `--agent-id` or `--config-path` argument must be specified.')
    if agent_id is not None:
        get_logs(agent_id=agent_id, tail=tail)
    if config_path is not None:
        config = VFLConfig.load_and_validate(config_path)
        logger.info(f'Test-report-logs path: {config.common.reports_export_folder}')


@cli.command()
@click.option('--config-path', type=str, required=True)
def predict():
    click.echo('Run VFL predictions')
    # TODO
    raise NotImplementedError


@cli.group()
def report():
    """ Experimental report command group. """
    click.echo('Get report on the experiment')


@report.command()
@click.option('--config-path', type=str, required=True)
@click.option(
    '--report-save-dir',
    type=str,
    default=None,
    help='If not specified, uses `config.common.reports_export_folder` path to save results'
)
def export(config_path, report_save_dir):
    """
    Export MlFlow results into CSV format.

    :param config_path: Path to the VFL config file
    :param report_save_dir: Directory to save report. If not specified `config.common.reports_export_folder` is used
    """
    config = VFLConfig.load_and_validate(config_path)
    file_name = f'mlflow-experiment-{config.common.experiment_label}.csv'
    if report_save_dir is not None:
        os.makedirs(os.path.dirname(report_save_dir), exist_ok=True)
        save_dir = os.path.abspath(report_save_dir)
    else:
        save_dir = os.path.abspath(config.common.reports_export_folder)
    _mlflow.set_tracking_uri(f"http://{config.prerequisites.mlflow_host}:{config.prerequisites.mlflow_port}")
    response_mlflow = _mlflow.search_runs(experiment_names=[config.common.experiment_label])
    assert response_mlflow.shape[0] != 0, 'MlFlow returned empty dataframe. Skipping saving report'
    full_save_path = os.path.join(save_dir, file_name)
    response_mlflow.to_csv(full_save_path)
    logger.info(f'Saved MlFlow report to: {full_save_path}')


@report.command()
@click.option('--config-path', type=str, required=True)
def mlflow(config_path):
    """
    Print URI of the MlFlow experiment.

    :param config_path: Path to the VFL config file
    """
    config = VFLConfig.load_and_validate(config_path)
    mlflow_uri = f"http://{config.prerequisites.mlflow_host}:{config.prerequisites.mlflow_port}"
    _mlflow.set_tracking_uri(mlflow_uri)
    response_mlflow = _mlflow.get_experiment_by_name(config.common.experiment_label)
    if response_mlflow is None:
        logger.info(f'Experiment {config.common.experiment_label} not found. MlFlow URI: {mlflow_uri}')
    else:
        logger.info(f'MlFLow URI of the current experiment: {mlflow_uri}/#/experiments/{response_mlflow.experiment_id}')


@report.command()
@click.option('--config-path', type=str, required=True)
def prometheus(config_path):
    """
    Print Prometheus and Grafana URIs.

    :param config_path: Path to the VFL config file
    """
    config = VFLConfig.load_and_validate(config_path)
    logger.info(
        'Prometheus UI is available at: '
        f'http://{config.prerequisites.prometheus_host}:{config.prerequisites.prometheus_port}. '
        'Additionally, you can explore results in Grafana: '
        f'http://{config.prerequisites.prometheus_host}:{config.prerequisites.grafana_port}'
    )


if __name__ == "__main__":
    cli()
