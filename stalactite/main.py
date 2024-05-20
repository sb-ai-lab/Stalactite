"""Stalactite CLI.

This script implements main Stalactite command line interface commands,
allowing launch of the VFL experiments in single-node or multi-node mode
using the configuration file in the YAML format.
"""
import enum
import logging
import os
from pathlib import Path

import click
import mlflow as _mlflow
from docker.errors import APIError

from docker import APIClient
from stalactite.configs import VFLConfig
from stalactite.helpers import global_logging
from stalactite.utils_main import (
    BASE_CONTAINER_LABEL,
    BASE_MASTER_CONTAINER_NAME,
    BASE_ARBITER_CONTAINER_NAME,
    BASE_MEMBER_CONTAINER_NAME,
    KEY_CONTAINER_LABEL,
    get_env_vars,
    get_logs,
    get_status,
    is_test_environment,
    run_subprocess_command,
    stop_containers,
    start_distributed_agent,
    start_multiprocess_agents,
    run_local_experiment,
    create_external_network,
)

logging.getLogger('git').setLevel(logging.ERROR)
logging.getLogger('docker').setLevel(logging.ERROR)
logging.getLogger('fsspec').setLevel(logging.ERROR)
logger = logging.getLogger(__name__)
global_logging(logging_level=logging.DEBUG)


class PrerequisitesGroup(str, enum.Enum):
    mlflow = 'mlflow'
    monitoring = 'monitoring'


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
@click.option(
    "--group",
    type=click.Choice(PrerequisitesGroup.__members__),
    required=False,
    help="Which prerequisites to start. One of: `mlflow`, `monitoring`. If no group is passes, starts both groups."
)
def start(config_path, detached, group):
    """
    Start containers with prerequisites (MlFlow, Postgresql, Prometheus, Grafana).
    Group `mlflow` starts two containers: MlFlow and Postgresql for experimental metrics reporting.
    Group `monitoring` launches Prometheus and Grafana, which scrap runtime metrics and visualize it via Grafana
    dashboards.

    Note! `monitoring` group must be started at the same host as the VFL master to be able to see the
    targets from which the metrics are scraped.
    """
    if group is not None:
        group = [group]
    else:
        group = list(PrerequisitesGroup.__members__)

    logger.info(f"Starting prerequisites containers ({group})")
    config = VFLConfig.load_and_validate(config_path)
    env_vars = get_env_vars(config)
    create_external_network()
    for group_name in group:
        command = f"{config.docker.docker_compose_command} -f docker-compose-{group_name}.yml"
        run_subprocess_command(
            command=command + " up" + (" -d" if detached else "") + " --build",
            logger_err_info="Failed build process",
            cwd=config.docker.docker_compose_path,
            env=env_vars,
            shell=True,
        )
        if detached:
            if group_name == PrerequisitesGroup.monitoring:
                logger.info(
                    f"Prometheus port: {config.prerequisites.prometheus_port}\n"
                    f"Grafana port: {config.prerequisites.grafana_port}"
                )
            else:
                logger.info(
                    f"MlFlow port: {config.prerequisites.mlflow_port}\n"
                )


@prerequisites.command()
@click.option(
    "--config-path", type=str, required=True, help="Absolute path to the configuration file in `YAML` format."
)
@click.option("--remove", is_flag=True, show_default=True, default=False, help="Delete created containers, networks.")
@click.option("--remove-volumes", is_flag=True, show_default=True, default=False, help="Delete created volumes.")
@click.option(
    "--group",
    type=click.Choice(PrerequisitesGroup.__members__),
    required=False,
    help="Which prerequisites to stop. One of: `mlflow`, `monitoring`. If no group is passes, stops both groups."
)
def stop(config_path, remove, remove_volumes, group):
    """
    Stop prerequisites containers (and remove all defined in docker-compose.yml networks and images[, and volumes]).
    Group `mlflow` stops both MlFlow and Postgresql.
    Group `monitoring` stops Prometheus and Grafana containers.
    """
    if group is not None:
        group = [group]
    else:
        group = list(PrerequisitesGroup.__members__)

    logger.info(f"Stopping prerequisites containers ({group})")
    config = VFLConfig.load_and_validate(config_path)
    env_vars = get_env_vars(config)
    for group_name in group:
        command = f"{config.docker.docker_compose_command} -f docker-compose-{group_name}.yml"
        run_subprocess_command(
            command=command + " stop",
            logger_err_info="Failed stopping prerequisites containers",
            cwd=config.docker.docker_compose_path,
            env=env_vars,
            shell=True,
        )
        logger.info(f"Successfully stopped prerequisite containers ({group_name})")
        if remove:
            run_subprocess_command(
                command=command + " down" + (" -v" if remove_volumes else ""),
                logger_err_info="Failed releasing resources",
                cwd=config.docker.docker_compose_path,
                env=env_vars,
                shell=True,
            )
            logger.info(f"Successfully teared down prerequisite resources ({group_name})")


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
)
@click.option("-d", "--detached", is_flag=True, show_default=True, default=False, help="Run non-blocking start.")
@click.option("--infer", is_flag=True, show_default=True, default=False, help="Run distributed inference.")
@click.pass_context
def start(ctx, config_path, rank, detached, infer):
    """
    Start a VFL-distributed Member in an isolated container.

    :param ctx: Click context
    :param config_path: Absolute path to the configuration file in `YAML` format
    :param rank: Rank of the member used for correct data loading
    :param detached: Run non-blocking start
    :param infer: Run member in an inference mode (distributed VFL prediction)
    """
    start_distributed_agent(
        config_path=config_path, role='member', rank=rank, infer=infer, detached=detached, ctx=ctx
    )


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
        stop_containers(client, containers, leave_containers=leave_containers)
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
        containers_label=f"{KEY_CONTAINER_LABEL}={container_label}",
        docker_client=client,
    )


@member.command()
@click.option("--agent-id", type=str, default=None, help="ID of the agent`s container.")
@click.option("--rank", type=str, default=None, help="Rank of the member.")
@click.option("--follow", is_flag=True, show_default=True, default=False, help="Follow log output.")
@click.option("--tail", type=str, default="all", help="Number of lines to show from the end of the logs.")
@click.option("--infer", is_flag=True, show_default=True, default=False, help="Show logs of the distributed inference.")
@click.pass_context
def logs(ctx, agent_id, rank, follow, tail, infer):
    """
    Retrieve members` containers logs present at the time of execution.

    :param ctx: Click context
    :param agent_id: ID of the agent`s container
    :param rank: Rank of the member
    :param follow: Follow log output
    :param tail: Number of lines to show from the end of the logs. Can be `all` or a positive integer
    :param infer: Get logs of the inference mode (distributed VFL prediction) container
    """
    if agent_id is None and rank is None:
        raise SyntaxError("Either `--agent-id` or `--rank` must be passed.")
    if agent_id is not None:
        container_name_or_id = agent_id
    else:
        container_name_or_id = ctx.obj["member_container_name"](rank) + ("-predict" if infer else "")
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
@click.option("--infer", is_flag=True, show_default=True, default=False, help="Run distributed inference.")
@click.pass_context
def start(ctx, config_path, detached, infer):
    """
    Start VFL Master in an isolated container.

    :param ctx: Click context
    :param config_path: Absolute path to the configuration file in `YAML` format
    :param detached: Run non-blocking start
    :param infer: Run master in an inference mode (distributed VFL prediction)
    """
    start_distributed_agent(
        config_path=config_path, role='master', infer=infer, detached=detached, ctx=ctx
    )


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
    for name in [ctx.obj["master_container_name"], ctx.obj["master_container_name"] + "-predict"]:
        try:
            containers = client.containers(all=True, filters={"name": name})
            if len(containers) < 1:
                logger.warning("Found 0 containers. Skipping.")
                return
            stop_containers(client, containers, leave_containers=leave_containers)
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
        agent_id=None, containers_label=f"{KEY_CONTAINER_LABEL}={container_label}", docker_client=client
    )


@master.command()
@click.option("--follow", is_flag=True, show_default=True, default=False, help="Follow log output.")
@click.option("--tail", type=str, default="all", help="Number of lines to show from the end of the logs.")
@click.option("--infer", is_flag=True, show_default=True, default=False, help="Show logs of the distributed inference.")
@click.pass_context
def logs(ctx, follow, tail, infer):
    """
    Retrieve master`s container logs present at the time of execution.

    :param ctx: Click context
    :param follow: Follow log output
    :param tail: Number of lines to show from the end of the logs. Can be `all` or a positive integer
    :param infer: Get logs of the inference mode (distributed VFL prediction) container
    """
    client = APIClient()
    get_logs(
        agent_id=ctx.obj["master_container_name"] + ("-predict" if infer else ""),
        tail=tail,
        follow=follow,
        docker_client=client,
    )


@cli.group()
@click.pass_context
def arbiter(ctx):
    """Distributed VFL arbiter management command group."""
    click.echo("Manage distributed (multi-node) arbiter")
    ctx.obj = dict()
    postfix = "-distributed"
    ctx.obj["arbiter_container_label"] = BASE_CONTAINER_LABEL + postfix + "-arbiter"
    ctx.obj["arbiter_container_name"] = BASE_ARBITER_CONTAINER_NAME + postfix


@arbiter.command()
@click.option(
    "--config-path", type=str, required=True, help="Absolute path to the configuration file in `YAML` format."
)
@click.option("-d", "--detached", is_flag=True, show_default=True, default=False, help="Run non-blocking start.")
@click.option("--infer", is_flag=True, show_default=True, default=False, help="Run distributed inference.")
@click.pass_context
def start(ctx, config_path, detached, infer):
    """
    Start VFL Master in an isolated container.

    :param ctx: Click context
    :param config_path: Absolute path to the configuration file in `YAML` format
    :param detached: Run non-blocking start
    :param infer: Run arbiter in an inference mode (distributed VFL prediction)
    """
    start_distributed_agent(
        config_path=config_path, role='arbiter', infer=infer, detached=detached, ctx=ctx
    )


@arbiter.command()
@click.option(
    "--leave-containers", is_flag=True, show_default=False, default=False, help="Retain created agents containers."
)
@click.pass_context
def stop(ctx, leave_containers):
    """
    Stop VFL arbiter container.

    :param ctx: Click context
    :param leave_containers: Retain created arbiter container
    """
    logger.info("Stopping arbiter container")
    client = ctx.obj.get("client", APIClient())
    for name in [ctx.obj["arbiter_container_name"], ctx.obj["arbiter_container_name"] + "-predict"]:
        try:
            containers = client.containers(all=True, filters={"name": name})
            if len(containers) < 1:
                logger.warning("Found 0 containers. Skipping.")
                return
            stop_containers(client, containers, leave_containers=leave_containers)
        except APIError as exc:
            logger.error("Error while stopping (and removing) arbiter container", exc_info=exc)


@arbiter.command()
@click.pass_context
def status(ctx):
    """
    Get the status of created arbiter`s container.

    :param ctx: Click context
    """
    container_label = ctx.obj["arbiter_container_label"]
    client = ctx.obj.get("client", APIClient())
    get_status(
        agent_id=None, containers_label=f"{KEY_CONTAINER_LABEL}={container_label}", docker_client=client
    )


@arbiter.command()
@click.option("--follow", is_flag=True, show_default=True, default=False, help="Follow log output.")
@click.option("--tail", type=str, default="all", help="Number of lines to show from the end of the logs.")
@click.option("--infer", is_flag=True, show_default=True, default=False, help="Show logs of the distributed inference.")
@click.pass_context
def logs(ctx, follow, tail, infer):
    """
    Retrieve master`s container logs present at the time of execution.

    :param ctx: Click context
    :param follow: Follow log output
    :param tail: Number of lines to show from the end of the logs. Can be `all` or a positive integer
    :param infer: Get logs of the inference mode (distributed VFL prediction) container
    """
    client = APIClient()
    get_logs(
        agent_id=ctx.obj["arbiter_container_name"] + ("-predict" if infer else ""),
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
    if ctx.obj["multi_process"] and not ctx.obj["single_process"]:
        client = ctx.obj.get("client", APIClient())
        start_multiprocess_agents(config_path, client=client, test=_test)
    elif ctx.obj["single_process"] and not ctx.obj["multi_process"]:
        run_local_experiment(config_path=config_path)


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
            stop_containers(client, containers, leave_containers=leave_containers)
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
            client = ctx.obj.get("client", APIClient())
            try:
                container_label = BASE_CONTAINER_LABEL + ("-test" if _test else "")
                containers = client.containers(all=True, filters={"label": f"{KEY_CONTAINER_LABEL}={container_label}"})
                stop_containers(client, containers, leave_containers=False)
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
    get_status(agent_id=agent_id, containers_label=f"{KEY_CONTAINER_LABEL}={container_label}")


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
def predict(multi_process, single_process, config_path):
    click.echo("Run VFL predictions")
    if multi_process and not single_process:
        client = APIClient()
        start_multiprocess_agents(
            config_path=config_path,
            client=client,
            test=False,
            is_infer=True,
        )
    elif single_process and not multi_process:
        run_local_experiment(config_path, is_infer=True)


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
