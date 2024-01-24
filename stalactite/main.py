"""Stalactite CLI.

This script implements main Stalactite command line interface commands,
allowing launch of the VFL experiments in single-node or multi-node mode
using the configuration file in the YAML format.
"""

import logging
import os
import threading
from pathlib import Path
from threading import Thread

import click
import mlflow as _mlflow
from docker.errors import APIError

from docker import APIClient
from stalactite.base import PartyMember
from stalactite.communications.local import (
    LocalMasterPartyCommunicator,
    LocalMemberPartyCommunicator,
)
from stalactite.configs import VFLConfig, raise_path_not_exist
from stalactite.data_utils import get_party_master, get_party_member
from stalactite.utils_main import (
    BASE_CONTAINER_LABEL,
    BASE_IMAGE_TAG,
    BASE_MASTER_CONTAINER_NAME,
    BASE_MEMBER_CONTAINER_NAME,
    KEY_CONTAINER_LABEL,
    PREREQUISITES_NETWORK,
    build_base_image,
    get_env_vars,
    get_logs,
    get_status,
    is_test_environment,
    run_subprocess_command,
    stop_containers,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
sh = logging.StreamHandler()
sh.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
logger.addHandler(sh)


@click.group()
def cli():
    """Main stalactite CLI command group."""
    click.echo("Stalactite module API")


@cli.group()
def prerequisites():
    """Prerequisites management command group."""
    click.echo("Manage experimental prerequisites")


@prerequisites.command()
@click.option(
    "--config-path", type=str, required=True, help="Absolute path to the configuration file in `YAML` format."
)
@click.option(
    "-d", "--detached", is_flag=True, show_default=True, default=False, help="Run non-blocking start in detached mode."
)
def start(config_path, detached):
    """
    Start containers with prerequisites (MlFlow, Prometheus, Grafana).

    :param config_path: Absolute path to the configuration file in `YAML` format
    :param detached: Run non-blocking start in detached mode
    """
    logger.info("Starting prerequisites containers")
    config = VFLConfig.load_and_validate(config_path)
    env_vars = get_env_vars(config)
    command = f"{config.docker.docker_compose_command}"
    run_subprocess_command(
        command=command + " up" + (" -d" if detached else "") + " --build",
        logger_err_info="Failed build process",
        cwd=config.docker.docker_compose_path,
        env=env_vars,
        shell=True,
    )
    if detached:
        logger.info(
            f"MlFlow port: {config.prerequisites.mlflow_port}\n"
            f"Prometheus port: {config.prerequisites.prometheus_port}\n"
            f"Grafana port: {config.prerequisites.grafana_port}"
        )


@prerequisites.command()
@click.option(
    "--config-path", type=str, required=True, help="Absolute path to the configuration file in `YAML` format."
)
@click.option("--remove", is_flag=True, show_default=True, default=False, help="Delete created containers, networks.")
@click.option("--remove-volumes", is_flag=True, show_default=True, default=False, help="Delete created volumes.")
def stop(config_path, remove, remove_volumes):
    """
    Stop prerequisites containers (and remove all defined in docker-compose.yml networks and images[, and volumes]).

    :param config_path: Absolute path to the configuration file in `YAML` format
    :param remove: Delete created containers, networks
    :param remove_volumes: Delete created volumes
    """
    logger.info("Stopping prerequisites containers")
    config = VFLConfig.load_and_validate(config_path)
    env_vars = get_env_vars(config)
    command = f"{config.docker.docker_compose_command}"
    run_subprocess_command(
        command=command + " stop",
        logger_err_info="Failed stopping prerequisites containers",
        cwd=config.docker.docker_compose_path,
        env=env_vars,
        shell=True,
    )
    logger.info(f"Successfully stopped prerequisite containers")
    if remove:
        run_subprocess_command(
            command=command + " down" + (" -v" if remove_volumes else ""),
            logger_err_info="Failed releasing resources",
            cwd=config.docker.docker_compose_path,
            env=env_vars,
            shell=True,
        )
        logger.info(f"Successfully teared down prerequisite resources")


@cli.group()
@click.pass_context
def member(ctx):
    """Distributed VFL member management command group."""
    click.echo("Manage distributed (multi-node) members")
    ctx.obj = dict()
    postfix = "-distributed"
    ctx.obj["member_container_label"] = BASE_CONTAINER_LABEL + postfix + "-member"
    ctx.obj["member_container_name"] = lambda rank: BASE_MEMBER_CONTAINER_NAME + f"-{rank}" + postfix


@member.command()
@click.option(
    "--config-path", type=str, required=True, help="Absolute path to the configuration file in `YAML` format."
)
@click.option(
    "--rank", type=int, help="Rank of the member used for correct data loading."
)  # TODO remove after refactoring of the preprocessing
@click.option("-d", "--detached", is_flag=True, show_default=True, default=False, help="Run non-blocking start.")
@click.pass_context
def start(ctx, config_path, rank, detached):
    """
    Start a VFL-distributed Member in an isolated container.

    :param ctx: Click context
    :param config_path: Absolute path to the configuration file in `YAML` format
    :param rank: Rank of the member used for correct data loading
    :param detached: Run non-blocking start
    """
    config = VFLConfig.load_and_validate(config_path)
    logger.info("Starting member for distributed experiment")
    client = APIClient()
    logger.info("Building an image for the member container. If build for the first time, it may take a while...")
    try:
        build_base_image(client, logger=logger)

        logger.info(f"Starting gRPC member-{rank} container")

        configs_path = os.path.dirname(os.path.abspath(config_path))

        raise_path_not_exist(config.data.host_path_data_dir)
        raise_path_not_exist(configs_path)

        data_path = os.path.abspath(config.data.host_path_data_dir)

        member_container = client.create_container(
            image=BASE_IMAGE_TAG,
            command=["python", "run_grpc_member.py", "--config-path", f"{os.path.abspath(config_path)}"],
            detach=True,
            environment={"RANK": rank},
            labels={KEY_CONTAINER_LABEL: ctx.obj["member_container_label"]},
            volumes=[f"{data_path}", f"{configs_path}"],
            host_config=client.create_host_config(
                binds=[
                    f"{configs_path}:{configs_path}:rw",
                    f"{data_path}:{data_path}:rw",
                ],
            ),
            name=ctx.obj["member_container_name"](rank),
        )
        logger.info(f'Member {rank} container id: {member_container.get("Id")}')
        client.start(container=member_container.get("Id"))
        if not detached:
            output = client.attach(member_container, stream=True, logs=True)
            try:
                for log in output:
                    print(log.decode().strip())
            except KeyboardInterrupt:
                client.stop(member_container)  # TODO question: stop if interrupted?
    except APIError as exc:
        logger.error(f"Error while starting member-{rank} container", exc_info=exc)
        raise


@member.command()
@click.option(
    "--leave-containers", is_flag=True, show_default=False, default=False, help="Retain created agents containers."
)
@click.pass_context
def stop(ctx, leave_containers):
    """
    Stop distributed VFL members containers started on current host.

    :param ctx: Click context
    :param leave_containers: Retain created agents containers
    """
    logger.info("Stopping members` containers")
    client = APIClient()
    try:
        container_label = ctx.obj["member_container_label"]
        containers = client.containers(all=True, filters={"label": f"{KEY_CONTAINER_LABEL}={container_label}"})
        stop_containers(client, containers, leave_containers=leave_containers, logger=logger)
    except APIError as exc:
        logger.error("Error while stopping (and removing) containers", exc_info=exc)


@member.command()
@click.option("--agent-id", type=str, default=None, help="ID of the agent`s container.")
@click.option("--rank", type=str, default=None, help="Rank of the member.")
@click.pass_context
def status(ctx, agent_id, rank):
    """
    Get the status of created members` containers.
    If the `agent-id` and `rank` are not passed, all the created members` containers statuses will be returned.

    :param ctx: Click context
    :param agent_id: ID of the agent`s container
    :param rank: Rank of the member
    """
    client = ctx.obj.get("client", APIClient())
    container_label = ctx.obj["member_container_label"]
    if agent_id is None and rank is None:
        get_status(
            agent_id=None,
            logger=logger,
            containers_label=f"{KEY_CONTAINER_LABEL}={container_label}",
            docker_client=client,
        )
        return
    elif agent_id is not None:
        container_name_or_id = agent_id
    else:
        container_name_or_id = ctx.obj["member_container_name"](rank)
    get_status(
        agent_id=container_name_or_id,
        logger=logger,
        containers_label=f"{KEY_CONTAINER_LABEL}={container_label}",
        docker_client=client,
    )


@member.command()
@click.option("--agent-id", type=str, default=None, help="ID of the agent`s container.")
@click.option("--rank", type=str, default=None, help="Rank of the member.")  # TODO rm after refactor?
@click.option("--follow", is_flag=True, show_default=True, default=False, help="Follow log output.")
@click.option("--tail", type=str, default="all", help="Number of lines to show from the end of the logs.")
@click.pass_context
def logs(ctx, agent_id, rank, follow, tail):
    """
    Retrieve members` containers logs present at the time of execution.

    :param ctx: Click context
    :param agent_id: ID of the agent`s container
    :param rank: Rank of the member
    :param follow: Follow log output
    :param tail: Number of lines to show from the end of the logs. Can be `all` or a positive integer
    """
    if agent_id is None and rank is None:
        raise SyntaxError("Either `--agent-id` or `--rank` must be passed.")
    if agent_id is not None:
        container_name_or_id = agent_id
    else:
        container_name_or_id = ctx.obj["member_container_name"](rank)
    client = APIClient()
    get_logs(
        agent_id=container_name_or_id,
        tail=tail,
        follow=follow,
        docker_client=client,
    )


@cli.group()
@click.pass_context
def master(ctx):
    """Distributed VFL master management command group."""
    click.echo("Manage distributed (multi-node) master")
    ctx.obj = dict()
    postfix = "-distributed"
    ctx.obj["master_container_label"] = BASE_CONTAINER_LABEL + postfix + "-master"
    ctx.obj["master_container_name"] = BASE_MASTER_CONTAINER_NAME + postfix


@master.command()
@click.option(
    "--config-path", type=str, required=True, help="Absolute path to the configuration file in `YAML` format."
)
@click.option("-d", "--detached", is_flag=True, show_default=True, default=False, help="Run non-blocking start.")
@click.pass_context
def start(ctx, config_path, detached):
    """
    Start VFL Master in an isolated container.

    :param ctx: Click context
    :param config_path: Absolute path to the configuration file in `YAML` format
    :param detached: Run non-blocking start
    """
    config = VFLConfig.load_and_validate(config_path)
    logger.info("Starting master for distributed experiment")
    client = APIClient()
    logger.info("Building an image for the master container. If build for the first time, it may take a while...")
    build_base_image(client, logger=logger)

    raise_path_not_exist(config.data.host_path_data_dir)

    data_path = os.path.abspath(config.data.host_path_data_dir)
    configs_path = os.path.dirname(os.path.abspath(config_path))

    raise_path_not_exist(configs_path)

    try:
        logger.info("Starting gRPC master container")

        master_container = client.create_container(
            image=BASE_IMAGE_TAG,
            command=["python", "run_grpc_master.py", "--config-path", f"{os.path.abspath(config_path)}"],
            detach=True,
            labels={KEY_CONTAINER_LABEL: ctx.obj["master_container_label"]},
            volumes=[f"{data_path}", f"{configs_path}"],
            host_config=client.create_host_config(
                binds=[
                    f"{configs_path}:{configs_path}:rw",
                    f"{data_path}:{data_path}:rw",
                ],
                port_bindings={config.grpc_server.port: config.grpc_server.port},
            ),
            ports=[config.grpc_server.port],
            name=ctx.obj["master_container_name"],
        )
        client.start(container=master_container.get("Id"))
        logger.info(f'Master container id: {master_container.get("Id")}')
        if not detached:
            output = client.attach(master_container, stream=True, logs=True)
            try:
                for log in output:
                    print(log.decode().strip())
            except KeyboardInterrupt:
                client.stop(master_container)  # TODO question: stop if interrupted?

    except APIError as exc:
        logger.error("Error while starting master container", exc_info=exc)
        raise


@master.command()
@click.option(
    "--leave-containers", is_flag=True, show_default=False, default=False, help="Retain created agents containers."
)
@click.pass_context
def stop(ctx, leave_containers):
    """
    Stop VFL master container.

    :param ctx: Click context
    :param leave_containers: Retain created master container
    """
    logger.info("Stopping master container")
    client = ctx.obj.get("client", APIClient())
    try:
        containers = client.containers(all=True, filters={"name": ctx.obj["master_container_name"]})
        if len(containers) < 1:
            logger.warning("Found 0 containers. Skipping.")
            return
        stop_containers(client, containers, leave_containers=leave_containers, logger=logger)
    except APIError as exc:
        logger.error("Error while stopping (and removing) master container", exc_info=exc)


@master.command()
@click.pass_context
def status(ctx):
    """
    Get the status of created master`s container.

    :param ctx: Click context
    """
    container_label = ctx.obj["master_container_label"]
    client = ctx.obj.get("client", APIClient())
    get_status(
        agent_id=None, logger=logger, containers_label=f"{KEY_CONTAINER_LABEL}={container_label}", docker_client=client
    )


@master.command()
@click.option("--follow", is_flag=True, show_default=True, default=False, help="Follow log output.")
@click.option("--tail", type=str, default="all", help="Number of lines to show from the end of the logs.")
@click.pass_context
def logs(ctx, follow, tail):
    """
    Retrieve master`s container logs present at the time of execution.

    :param ctx: Click context
    :param follow: Follow log output
    :param tail: Number of lines to show from the end of the logs. Can be `all` or a positive integer
    """
    client = APIClient()
    get_logs(
        agent_id=ctx.obj["master_container_name"],
        tail=tail,
        follow=follow,
        docker_client=client,
    )


@cli.group()
@click.option(
    "--single-process",
    is_flag=True,
    show_default=True,
    default=False,
    help="Run single-node single-process (multi-thread) test.",
)
@click.option(
    "--multi-process",
    is_flag=True,
    show_default=True,
    default=False,
    help="Run single-node multi-process (dockerized) test.",
)
@click.pass_context
def local(ctx, single_process, multi_process):
    """
    Local experiments (multi-process / single process) mode command group.

    :param ctx: Click context
    :param single_process: Run single process experiment
    :param multi_process: Run multiple process (dockerized) experiment
    """
    ctx.obj = dict()
    if multi_process and not single_process:
        click.echo("Multiple-process single-node mode")
        ctx.obj["client"] = APIClient()
    elif single_process and not multi_process:
        click.echo("Multiple-threads single-node mode")
    else:
        raise SyntaxError("Either `--single-process` or `--multi-process` flag can be set.")
    ctx.obj["multi_process"] = multi_process
    ctx.obj["single_process"] = single_process


@local.command()
@click.option(
    "--config-path", type=str, required=True, help="Absolute path to the configuration file in `YAML` format."
)
@click.pass_context
def start(ctx, config_path):
    """
    Start local experiment.
    For a multiprocess mode build and start VFL master and members containers.
    Single-process mode starts a python process with each agent has a thread.

    :param ctx: Click context
    :param config_path: Absolute path to the configuration file in `YAML` format
    """
    _test = is_test_environment()
    config = VFLConfig.load_and_validate(config_path)
    if ctx.obj["multi_process"] and not ctx.obj["single_process"]:
        logger.info("Starting multi-process single-node experiment")
        client = ctx.obj.get("client", APIClient())
        logger.info("Building an image of the agent. If build for the first time, it may take a while...")
        build_base_image(client, logger=logger)

        if networks := client.networks(names=[PREREQUISITES_NETWORK]):
            network = networks.pop()["Name"]
        else:
            network = PREREQUISITES_NETWORK
            client.create_network(network)
        networking_config = client.create_networking_config({network: client.create_endpoint_config()})

        raise_path_not_exist(config.data.host_path_data_dir)

        data_path = os.path.abspath(config.data.host_path_data_dir)
        configs_path = os.path.dirname(os.path.abspath(config_path))

        raise_path_not_exist(configs_path)

        volumes = [f"{data_path}", f"{configs_path}"]
        mounts_host_config = client.create_host_config(
            binds=[
                f"{configs_path}:{configs_path}:rw",
                f"{data_path}:{data_path}:rw",
            ]
        )

        container_label = BASE_CONTAINER_LABEL + ("-test" if _test else "")
        master_container_name = BASE_MASTER_CONTAINER_NAME + ("-test" if _test else "")
        try:
            logger.info("Starting gRPC master container")
            grpc_server_host = "grpc-master"

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
            client.start(container=master_container.get("Id"))

            for member_rank in range(config.common.world_size):
                logger.info(f"Starting gRPC member-{member_rank} container")
                member_container_name = BASE_MEMBER_CONTAINER_NAME + f"-{member_rank}" + ("-test" if _test else "")

                member_container = client.create_container(
                    image=BASE_IMAGE_TAG,
                    command=["python", "run_grpc_member.py", "--config-path", f"{os.path.abspath(config_path)}"],
                    detach=True,
                    environment={"RANK": member_rank, "GRPC_SERVER_HOST": grpc_server_host},
                    labels={KEY_CONTAINER_LABEL: container_label},
                    volumes=volumes,
                    host_config=mounts_host_config,
                    networking_config=networking_config,
                    name=member_container_name,
                )
                logger.info(f'Member {member_rank} container id: {member_container.get("Id")}')
                client.start(container=member_container.get("Id"))
        except APIError as exc:
            logger.error(
                "Error while agents containers launch. If the container name is in use, alternatively to renaming you "
                "can remove all containers from previous experiment by running "
                f'`stalactite {"test" if _test else "local"} --multi-process stop`',
                exc_info=exc,
            )
            raise
    elif ctx.obj["single_process"] and not ctx.obj["multi_process"]:
        master = get_party_master(config)
        members = [get_party_member(config, member_rank=rank) for rank in range(config.common.world_size)]
        shared_party_info = dict()
        if config.master.run_mlflow:
            _mlflow.set_tracking_uri(f"http://{config.prerequisites.mlflow_host}:{config.prerequisites.mlflow_port}")
            _mlflow.set_experiment(config.common.experiment_label)
            _mlflow.start_run()

        def local_master_main():
            logger.info("Starting thread %s" % threading.current_thread().name)
            comm = LocalMasterPartyCommunicator(
                participant=master, world_size=config.common.world_size, shared_party_info=shared_party_info
            )
            comm.run()
            logger.info("Finishing thread %s" % threading.current_thread().name)

        def local_member_main(member: PartyMember):
            logger.info("Starting thread %s" % threading.current_thread().name)
            comm = LocalMemberPartyCommunicator(
                participant=member, world_size=config.common.world_size, shared_party_info=shared_party_info
            )
            comm.run()
            logger.info("Finishing thread %s" % threading.current_thread().name)

        threads = [
            Thread(name=f"main_{master.id}", daemon=True, target=local_master_main),
            *(
                Thread(name=f"main_{member.id}", daemon=True, target=local_member_main, args=(member,))
                for member in members
            ),
        ]

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()
        if config.master.run_mlflow:
            _mlflow.end_run()


@local.command()
@click.option(
    "--leave-containers", is_flag=True, show_default=False, default=False, help="Retain created agents containers."
)
@click.pass_context
def stop(ctx, leave_containers):
    """
    Stop local experiment.
    For a multiprocess mode stop and remove containers of the VFL master and members.
    Does nothing for the single-process mode and is present for the CLI group coherency.

    :param ctx: Click context
    :param leave_containers: Retain created agents containers (available for `multi-process` mode only)
    """
    _test = is_test_environment()
    if ctx.obj["multi_process"] and not ctx.obj["single_process"]:
        logger.info("Stopping multi-process single-node")
        client = ctx.obj.get("client", APIClient())
        try:
            container_label = BASE_CONTAINER_LABEL + ("-test" if _test else "")
            containers = client.containers(all=True, filters={"label": f"{KEY_CONTAINER_LABEL}={container_label}"})
            stop_containers(client, containers, leave_containers=leave_containers, logger=logger)
        except APIError as exc:
            logger.error("Error while stopping (and removing) containers", exc_info=exc)
    else:
        logger.info("Nothing to stop")


@local.command()
@click.option("--agent-id", type=str, default=None, help="ID of the agents` container.")
@click.pass_context
def status(ctx, agent_id):
    """
    For the `multi-process` mode print status of the experimental container(s).
    If the `agent-id` is not passed, all the created members` containers statuses will be returned.

    Does nothing for the single-process mode and is present for the CLI group coherency.

    :param ctx: Click context
    :param agent_id: ID of the agent`s container
    """
    if ctx.obj["multi_process"] and not ctx.obj["single_process"]:
        _test = is_test_environment()
        container_label = BASE_CONTAINER_LABEL + ("-test" if _test else "")
        client = ctx.obj.get("client", APIClient())
        get_status(
            agent_id=agent_id,
            logger=logger,
            containers_label=f"{KEY_CONTAINER_LABEL}={container_label}",
            docker_client=client,
        )
    else:
        logger.info("Status in the single-process mode is not available")


@local.command()
@click.option("--agent-id", type=str, required=True, help="ID of the agents` container.")
@click.option("--follow", is_flag=True, show_default=True, default=False, help="Follow log output.")
@click.option("--tail", type=str, default="all", help="Number of lines to show from the end of the logs.")
@click.pass_context
def logs(ctx, agent_id, follow, tail):
    """
    Print logs of the experimental container.

    :param ctx: Click context
    :param agent_id: ID of the agent`s container
    :param follow: Follow log output
    :param tail: Number of lines to show from the end of the logs. Can be `all` or a positive integer
    """
    if ctx.obj["multi_process"] and not ctx.obj["single_process"]:
        client = ctx.obj.get("client", APIClient())
        get_logs(
            agent_id=agent_id,
            tail=tail,
            follow=follow,
            docker_client=client,
        )
    else:
        logger.info("Logs in the single-process mode are not available")


@cli.group()
@click.pass_context
@click.option(
    "--single-process",
    is_flag=True,
    show_default=True,
    default=False,
    help="Run single-node single-process (multi-thread) test.",
)
@click.option(
    "--multi-process",
    is_flag=True,
    show_default=True,
    default=False,
    help="Run single-node multi-process (dockerized) test.",
)
def test(ctx, multi_process, single_process):
    """
    Local tests (multi-process / single process) mode command group.

    :param ctx: Click context
    :param single_process: Run tests on single process experiment
    :param multi_process: Run tests on multiple process (dockerized) experiment
    """
    if multi_process and not single_process:
        click.echo("Manage multi-process single-node tests")
    elif single_process and not multi_process:
        click.echo("Manage single-process (multi-thread) single-node tests")
    else:
        raise SyntaxError("Either `--single-process` or `--multi-process` flag can be set.")
    ctx.obj = dict()
    ctx.obj["multi_process"] = multi_process
    ctx.obj["single_process"] = single_process


@test.command()
@click.pass_context
@click.option(
    "--config-path", type=str, required=True, help="Absolute path to the configuration file in `YAML` format."
)
def start(ctx, config_path):
    """
    Start local test experiment.
    For a multiprocess mode run integration tests on started VFL master and members containers.
    Single-process test will run tests within a python process.

    :param ctx: Click context
    :param config_path: Absolute path to the configuration file in `YAML` format
    """
    config = VFLConfig.load_and_validate(config_path)
    if ctx.obj["multi_process"] and not ctx.obj["single_process"]:
        test_group_name = "TestLocalGroupStart"
        report_file_name = f"{test_group_name}-log-{config.common.experiment_label}.jsonl"
        run_subprocess_command(
            command=f"python -m pytest --test_config_path {config_path} "
            f"tests/distributed_grpc/integration_test.py -k '{test_group_name}' -x "
            f"--report-log={os.path.join(config.common.reports_export_folder, report_file_name)} "
            "-W ignore::DeprecationWarning",
            logger_err_info="Failed running test",
            cwd=Path(__file__).parent.parent,
            shell=True,
        )
    else:
        report_file_name = f"local-tests-log-{config.common.experiment_label}.jsonl"
        run_subprocess_command(
            command=f"python -m pytest tests/test_local.py -x "
            f"--report-log={os.path.join(config.common.reports_export_folder, report_file_name)} "
            "-W ignore::DeprecationWarning --log-cli-level 20",
            logger_err_info="Failed running test",
            cwd=Path(__file__).parent.parent,
            shell=True,
        )


@test.command()
@click.pass_context
@click.option("--config-path", type=str, help="Absolute path to the configuration file in `YAML` format.")
@click.option(
    "--no-tests",
    is_flag=True,
    show_default=True,
    default=False,
    help="Remove test containers without launching Pytest.",
)
def stop(ctx, config_path, no_tests):
    """
    Stop local test experiment.
    For a multiprocess mode run integration tests while stopping VFL master and members containers.
    Does nothing for a single-process mode.

    :param ctx: Click context
    :param config_path: Absolute path to the configuration file in `YAML` format
    :param no_tests: Remove test containers without launching Pytest. Useful if `start` command did not succeed,
                     but containers have already been created
    """
    if config_path is None and not no_tests:
        raise SyntaxError("Specify `--config-path` or pass flag `--no-tests`")
    if no_tests:
        _test = 1
        if ctx.obj["multi_process"] and not ctx.obj["single_process"]:
            logger.info("Removing test containers")
            client = ctx.obj.get("client", APIClient())  # base_url='unix://var/run/docker.sock' TODO check
            try:
                container_label = BASE_CONTAINER_LABEL + ("-test" if _test else "")
                containers = client.containers(all=True, filters={"label": f"{KEY_CONTAINER_LABEL}={container_label}"})
                stop_containers(client, containers, leave_containers=False, logger=logger)
            except APIError as exc:
                logger.error("Error while stopping (and removing) containers", exc_info=exc)
            return
    config = VFLConfig.load_and_validate(config_path)
    if ctx.obj["multi_process"] and not ctx.obj["single_process"]:
        test_group_name = "TestLocalGroupStop"
        report_file_name = f"{test_group_name}-log-{config.common.experiment_label}.jsonl"
        run_subprocess_command(
            command=f"python -m pytest --test_config_path {config_path} "
            f"tests/distributed_grpc/integration_test.py -k '{test_group_name}' -x "
            f"--report-log={os.path.join(config.common.reports_export_folder, report_file_name)} "
            "-W ignore::DeprecationWarning",
            logger_err_info="Failed running test",
            cwd=Path(__file__).parent.parent,
            shell=True,
        )


@test.command()
@click.option("--agent-id", type=str, default=None, help="ID of the agents` container.")
def status(agent_id):
    """
    Print status of the experimental test container(s).
    If the `agent-id` is not passed, all the created on test containers` statuses will be returned.

    :param ctx: Click context
    :param agent_id: ID of the agents` container
    """
    _test = True
    container_label = BASE_CONTAINER_LABEL + ("-test" if _test else "")
    get_status(agent_id=agent_id, containers_label=f"{KEY_CONTAINER_LABEL}={container_label}", logger=logger)


@test.command()
@click.option("--agent-id", type=str, default=None, help="ID of the agents` container.")
@click.option("--tail", type=str, default="all", help="Number of lines to show from the end of the logs.")
@click.option("--config-path", type=str, default=None, help="Absolute path to the configuration file in `YAML` format.")
def logs(agent_id, config_path, tail):
    """
    Print logs of the experimental test container or return path to tests` logs.
    If the `agent-id` is passed, show container logs, otherwise, prints test report

    :param agent_id: ID of the agents` container
    :param config_path: Absolute path to the configuration file in `YAML` format
    :param tail: Number of lines to show from the end of the logs
    """
    if agent_id is None and config_path is None:
        raise SyntaxError("Either `--agent-id` or `--config-path` argument must be specified.")
    if agent_id is not None:
        get_logs(agent_id=agent_id, tail=tail)
    if config_path is not None:
        config = VFLConfig.load_and_validate(config_path)
        logger.info(f"Test-report-logs path: {config.common.reports_export_folder}")


@cli.command()
@click.option("--config-path", type=str, required=True)
def predict():
    click.echo("Run VFL predictions")
    # TODO
    raise NotImplementedError


@cli.group()
def report():
    """Experimental report command group."""
    click.echo("Get report on the experiment")


@report.command()
@click.option(
    "--config-path", type=str, required=True, help="Absolute path to the configuration file in `YAML` format."
)
@click.option("--report-save-dir", type=str, default=None, help="Directory to save the report.")
def export(config_path, report_save_dir):
    """
    Export MlFlow results into CSV format.

    :param config_path: Absolute path to the configuration file in `YAML` format
    :param report_save_dir: Directory to save the report. If not specified `config.common.reports_export_folder` is used
    """
    config = VFLConfig.load_and_validate(config_path)
    file_name = f"mlflow-experiment-{config.common.experiment_label}.csv"
    if report_save_dir is not None:
        os.makedirs(os.path.dirname(report_save_dir), exist_ok=True)
        save_dir = os.path.abspath(report_save_dir)
    else:
        save_dir = os.path.abspath(config.common.reports_export_folder)
    _mlflow.set_tracking_uri(f"http://{config.prerequisites.mlflow_host}:{config.prerequisites.mlflow_port}")
    response_mlflow = _mlflow.search_runs(experiment_names=[config.common.experiment_label])
    assert response_mlflow.shape[0] != 0, "MlFlow returned empty dataframe. Skipping saving report"
    full_save_path = os.path.join(save_dir, file_name)
    response_mlflow.to_csv(full_save_path)
    logger.info(f"Saved MlFlow report to: {full_save_path}")


@report.command()
@click.option(
    "--config-path", type=str, required=True, help="Absolute path to the configuration file in `YAML` format."
)
def mlflow(config_path):
    """
    Print URI of the MlFlow experiment.

    :param config_path: Absolute path to the configuration file in `YAML` format
    """
    config = VFLConfig.load_and_validate(config_path)
    mlflow_uri = f"http://{config.prerequisites.mlflow_host}:{config.prerequisites.mlflow_port}"
    _mlflow.set_tracking_uri(mlflow_uri)
    response_mlflow = _mlflow.get_experiment_by_name(config.common.experiment_label)
    if response_mlflow is None:
        logger.info(f"Experiment {config.common.experiment_label} not found. MlFlow URI: {mlflow_uri}")
    else:
        logger.info(f"MlFLow URI of the current experiment: {mlflow_uri}/#/experiments/{response_mlflow.experiment_id}")


@report.command()
@click.option(
    "--config-path", type=str, required=True, help="Absolute path to the configuration file in `YAML` format."
)
def prometheus(config_path):
    """
    Print Prometheus and Grafana URIs.

    :param config_path: Absolute path to the configuration file in `YAML` format
    """
    config = VFLConfig.load_and_validate(config_path)
    logger.info(
        "Prometheus UI is available at: "
        f"http://{config.prerequisites.prometheus_host}:{config.prerequisites.prometheus_port}. "
        "Additionally, you can explore results in Grafana: "
        f"http://{config.prerequisites.prometheus_host}:{config.prerequisites.grafana_port}"
    )


if __name__ == "__main__":
    cli()
