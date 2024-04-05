import logging
import os
import subprocess
import threading
from pathlib import Path
from typing import Optional, Any, List

from docker.errors import APIError, NotFound

from docker import APIClient

from stalactite.base import PartyMember
from stalactite.communications.local import ArbiteredLocalPartyCommunicator, LocalMasterPartyCommunicator, \
    LocalMemberPartyCommunicator
from stalactite.configs import VFLConfig, raise_path_not_exist
from stalactite.data_utils import get_party_arbiter, get_party_master, get_party_member
from stalactite.helpers import run_local_agents, reporting
from stalactite.ml.arbitered.base import Role

BASE_CONTAINER_LABEL = "grpc-experiment"
KEY_CONTAINER_LABEL = "container-g"
BASE_MASTER_CONTAINER_NAME = "master-agent-vfl"  # Do not change this value
BASE_ARBITER_CONTAINER_NAME = "arbiter-agent-vfl"  # Do not change this value
BASE_MEMBER_CONTAINER_NAME = "member-agent-vfl"
BASE_IMAGE_FILE = "grpc-base.dockerfile"
BASE_IMAGE_FILE_CPU = "grpc-base-cpu.dockerfile"
BASE_IMAGE_TAG = "grpc-base:latest"
PREREQUISITES_NETWORK = "prerequisites_vfl-network"  # Do not change this value

logging.getLogger('docker').setLevel(logging.ERROR)


def validate_int(value: Any, logger: logging.Logger = logging.getLogger("__main__")):
    try:
        value = int(value)
    except ValueError:
        logger.warning("Invalid argument. Must be positive int or `all`. Defaulting to `10`.")
        value = 10
    return value


def get_env_vars(config: VFLConfig) -> dict:
    env_vars = os.environ.copy()
    env_vars["MLFLOW_PORT"] = config.prerequisites.mlflow_port
    env_vars["PROMETHEUS_PORT"] = config.prerequisites.prometheus_port
    env_vars["GRAFANA_PORT"] = config.prerequisites.grafana_port
    env_vars["DOCKER_COMPOSE_PATH"] = config.docker.docker_compose_path

    return env_vars


def is_test_environment() -> bool:
    is_test = int(os.environ.get("STALACTITE_TEST_MODE", "0"))
    return bool(is_test)


def run_subprocess_command(
        command: str,
        logger_err_info: str,
        logger: logging.Logger = logging.getLogger("__main__"),
        **cmd_kwargs,
):
    process = subprocess.run(
        command,
        **cmd_kwargs,
    )
    try:
        process.check_returncode()
    except subprocess.CalledProcessError as exc:
        logger.error(logger_err_info, exc_info=exc)
        raise


def get_status(
        agent_id: Optional[str],
        containers_label: str,
        logger: logging.Logger,
        docker_client: APIClient = APIClient(),
):
    try:
        if agent_id is None:
            containers = docker_client.containers(all=True, filters={"label": containers_label})
        else:
            containers = docker_client.containers(all=True, filters={"id": agent_id})
        if not len(containers):
            logger.info("No containers to report status")
        for container in containers:
            logger.info(f'Container {container["Names"][0]} (id: {container["Id"][:12]}) status: {container["Status"]}')
    except APIError as exc:
        logger.error("Error checking container status", exc_info=exc)
        raise


def get_logs(
        agent_id: str,
        tail: str = "all",
        follow: bool = False,
        docker_client: APIClient = APIClient(),
        logger: logging.Logger = logging.getLogger("__main__"),
):
    if tail != "all":
        tail = validate_int(tail)
    try:
        logs_gen = docker_client.logs(container=agent_id, follow=follow, stream=True, tail=tail)
        for log in logs_gen:
            print(log.decode().strip())
    except NotFound:
        logger.info(f"No containers with name (id) {agent_id} were found.")
    except APIError as exc:
        logger.error("Error retrieving container logs", exc_info=exc)
        raise


def build_base_image(
        docker_client: APIClient,
        logger: logging.Logger = logging.getLogger("__main__"),
        use_gpu: bool = False
):
    image_file_name = BASE_IMAGE_FILE if use_gpu else BASE_IMAGE_FILE_CPU
    try:
        _logs = docker_client.build(
            path=str(Path(os.path.abspath(__file__)).parent.parent),
            tag=BASE_IMAGE_TAG,
            quiet=False,
            decode=True,
            nocache=False,
            dockerfile=os.path.join(Path(os.path.abspath(__file__)).parent.parent, "docker", image_file_name),
        )
        for log in _logs:
            logger.debug(log.get("stream", log.get("aux", {"aux": ""}).get("ID", "")).strip())
    except APIError as exc:
        logger.error("Error while building an image", exc_info=exc)
        raise


def stop_containers(
        docker_client: APIClient,
        containers: list,
        leave_containers: bool = False,
        logger: logging.Logger = logging.getLogger("__main__"),
):
    if not len(containers):
        logger.info("No containers to stop.")
    for container in containers:
        logger.info(f'Stopping {container["Id"]}')
        docker_client.stop(container)
        if not leave_containers:
            logger.info(f'Removing {container["Id"]}')
            docker_client.remove_container(container)


def create_and_start_container(
        client: APIClient,
        image: str,
        container_label: str,
        environment: dict,
        volumes: list,
        host_config: dict,
        container_name: str,
        role: str,
        config_path: str,
        hostname: Optional[str] = None,
        command: Optional[List[str]] = None,
        network_config: Optional[dict] = None,
        ports: Optional[list] = None,
        logger: logging.Logger = logging.getLogger("__main__"),
):
    logger.info(f"Starting gRPC {role} container")
    if command is None:
        command = [
            "python", "run_grpc_agent.py",
            "--config-path", f"{os.path.abspath(config_path)}",
            "--role", role
        ]
    container = client.create_container(
        image=image,
        command=command,
        detach=True,
        environment=environment,
        hostname=hostname,
        labels={KEY_CONTAINER_LABEL: container_label},
        volumes=volumes,
        host_config=host_config,
        networking_config=network_config,
        name=container_name,
        ports=ports
    )
    logger.info(f'{role.capitalize()} container ({container_name}) id: {container.get("Id")}')
    client.start(container=container.get("Id"))
    return container


def start_distributed_agent(
        config_path: str,
        role: str,
        infer: bool,
        detached: bool,
        ctx,
        logger: logging.Logger = logging.getLogger("__main__"),
        rank: int = None,
):
    config = VFLConfig.load_and_validate(config_path)
    if role == Role.arbiter and not config.grpc_arbiter.use_arbiter:
        raise ValueError('`config.grpc_arbiter.use_arbiter` is set to False, could not start the arbiter')
    if role == Role.member:
        assert rank is not None, "Member must be initialized with the `--rank` parameter"

    _logs_name = f"member-{rank}" if role == Role.member else role

    logger.info(f"Starting {role} for distributed experiment")
    client = APIClient()
    logger.info(f"Building an image for the {role} container. If build for the first time, it may take a while...")

    try:
        build_base_image(client, logger=logger, use_gpu=config.docker.use_gpu)

        configs_path = os.path.dirname(os.path.abspath(config_path))

        raise_path_not_exist(config.data.host_path_data_dir)
        raise_path_not_exist(configs_path)

        data_path = os.path.abspath(config.data.host_path_data_dir)
        model_path = os.path.abspath(config.vfl_model.vfl_model_path)

        if role == Role.master:
            if networks := client.networks(names=[PREREQUISITES_NETWORK]):
                network = networks.pop()["Name"]
            else:
                network = PREREQUISITES_NETWORK
                client.create_network(network)
            networking_config = client.create_networking_config({network: client.create_endpoint_config()})
        else:
            networking_config = None

        command = [
            "python",
            "run_grpc_agent.py",
            "--config-path", f"{os.path.abspath(config_path)}",
            "--role", f"{role}"]
        if infer:
            command.append("--infer")

        env_vars = {"RANK": rank} if role == Role.member else dict()
        if role == Role.master:
            port_binds = {config.grpc_server.port: config.grpc_server.port}
            ports = [config.grpc_server.port]
            name = ctx.obj["master_container_name"] + ("-predict" if infer else "")
        elif role == Role.arbiter:
            ports = [config.grpc_arbiter.port]
            port_binds = {config.grpc_arbiter.port: config.grpc_arbiter.port}
            name = ctx.obj["arbiter_container_name"] + ("-predict" if infer else "")
        else:
            port_binds, ports = dict(), list()
            name = ctx.obj["member_container_name"](rank) + ("-predict" if infer else "")

        container = create_and_start_container(
            network_config=networking_config,
            client=client,
            image=BASE_IMAGE_TAG,
            container_label=ctx.obj[f"{role}_container_label"],
            environment=env_vars,
            volumes=[f"{data_path}", f"{configs_path}", f"{model_path}"],
            host_config=client.create_host_config(
                binds=[
                    f"{configs_path}:{configs_path}:rw",
                    f"{data_path}:{data_path}:rw",
                    f"{model_path}:{model_path}:rw",
                ],
                port_bindings=port_binds,
            ),
            ports=ports,
            container_name=name,
            role=role,
            config_path=config_path,
            hostname=None,
            command=command,
            logger=logger,
        )

        if not detached:
            output = client.attach(container, stream=True, logs=True)
            try:
                for log in output:
                    print(log.decode().strip())
            except KeyboardInterrupt:
                client.stop(container)

    except APIError as exc:
        logger.error(f"Error while starting {_logs_name} container", exc_info=exc)
        raise


def start_multiprocess_agents(
        config_path: str,
        client: APIClient,
        test: bool,
        is_infer: bool = False,
        logger: logging.Logger = logging.getLogger("__main__")
):
    config = VFLConfig.load_and_validate(config_path)
    logger.info("Starting multi-process single-node experiment")
    logger.info("Building an image of the agent. If build for the first time, it may take a while...")
    build_base_image(client, logger=logger, use_gpu=config.docker.use_gpu)

    if networks := client.networks(names=[PREREQUISITES_NETWORK]):
        network = networks.pop()["Name"]
    else:
        network = PREREQUISITES_NETWORK
        client.create_network(network)
    networking_config = client.create_networking_config({network: client.create_endpoint_config()})

    raise_path_not_exist(config.data.host_path_data_dir)

    data_path = os.path.abspath(config.data.host_path_data_dir)
    configs_path = os.path.dirname(os.path.abspath(config_path))
    model_path = os.path.abspath(config.vfl_model.vfl_model_path)

    raise_path_not_exist(configs_path)

    volumes = [f"{data_path}", f"{configs_path}", f"{model_path}"]
    mounts_host_config = client.create_host_config(
        binds=[
            f"{configs_path}:{configs_path}:rw",
            f"{data_path}:{data_path}:rw",
            f"{model_path}:{model_path}:rw",
        ]
    )
    container_label = BASE_CONTAINER_LABEL
    master_container_name = BASE_MASTER_CONTAINER_NAME
    arbiter_container_name = BASE_ARBITER_CONTAINER_NAME

    if is_infer:
        master_container_name += "-predict"
        arbiter_container_name += "-predict"
    else:
        container_label += ("-test" if test else "")
        master_container_name += ("-test" if test else "")
        arbiter_container_name += ("-test" if test else "")

    grpc_arbiter_host = "grpc-arbiter"
    grpc_server_host = "grpc-master"

    try:
        if config.grpc_arbiter.use_arbiter:
            command_arbiter = [
                "python", "run_grpc_agent.py",
                "--config-path", f"{os.path.abspath(config_path)}",
                "--role", "arbiter",
                "--infer"
            ] if is_infer else None
            create_and_start_container(
                client=client,
                image=BASE_IMAGE_TAG,
                container_label=container_label,
                environment={},
                volumes=volumes,
                host_config=mounts_host_config,
                network_config=networking_config,
                container_name=arbiter_container_name,
                role='arbiter',
                config_path=config_path,
                hostname=grpc_arbiter_host,
                logger=logger,
                command=command_arbiter,
            )

        command_master = [
            "python", "run_grpc_agent.py",
            "--config-path", f"{os.path.abspath(config_path)}",
            "--infer",
            "--role", "master"
        ] if is_infer else None
        create_and_start_container(
            client=client,
            image=BASE_IMAGE_TAG,
            container_label=container_label,
            environment={"GRPC_ARBITER_HOST": grpc_arbiter_host},
            volumes=volumes,
            host_config=mounts_host_config,
            network_config=networking_config,
            container_name=master_container_name,
            role='master',
            config_path=config_path,
            hostname=grpc_server_host,
            logger=logger,
            command=command_master,
        )

        for member_rank in range(config.common.world_size):
            if is_infer:
                member_container_name = BASE_MEMBER_CONTAINER_NAME + f"-{member_rank}" + "-predict"
            else:
                member_container_name = BASE_MEMBER_CONTAINER_NAME + f"-{member_rank}" + ("-test" if test else "")

            member_command = [
                "python", "run_grpc_agent.py",
                "--config-path", f"{os.path.abspath(config_path)}",
                "--infer",
                "--role", "member"
            ] if is_infer else None

            create_and_start_container(
                client=client,
                image=BASE_IMAGE_TAG,
                container_label=container_label,
                environment={
                    "RANK": member_rank,
                    "GRPC_SERVER_HOST": grpc_server_host,
                    "GRPC_ARBITER_HOST": grpc_arbiter_host
                },
                volumes=volumes,
                host_config=mounts_host_config,
                network_config=networking_config,
                container_name=member_container_name,
                role='member',
                config_path=config_path,
                hostname=None,
                logger=logger,
                command=member_command,
            )

    except APIError as exc:
        logger.error(
            "Error while agents containers launch. If the container name is in use, alternatively to renaming you "
            "can remove all containers from previous experiment by running "
            f'`stalactite {"test" if test else "local"} --multi-process stop`',
            exc_info=exc,
        )
        raise


def run_local_experiment(config_path: str, is_infer: bool = False,
                         logger: logging.Logger = logging.getLogger("__main__")):
    config = VFLConfig.load_and_validate(config_path)
    arbiter = get_party_arbiter(config_path, is_infer=is_infer) if config.grpc_arbiter.use_arbiter else None
    master = get_party_master(config_path, is_infer=is_infer)
    members = [
        get_party_member(config_path, member_rank=rank, is_infer=is_infer)
        for rank in range(config.common.world_size)
    ]
    shared_party_info = dict()
    if config.grpc_arbiter.use_arbiter:
        master_communicator = ArbiteredLocalPartyCommunicator
        member_communicator = ArbiteredLocalPartyCommunicator
    else:
        master_communicator = LocalMasterPartyCommunicator
        member_communicator = LocalMemberPartyCommunicator

    def local_master_main():
        logger.info("Starting thread %s" % threading.current_thread().name)
        comm = master_communicator(
            participant=master,
            world_size=config.common.world_size,
            shared_party_info=shared_party_info,
            recv_timeout=config.master.recv_timeout,
        )
        comm.run()
        logger.info("Finishing thread %s" % threading.current_thread().name)

    def local_member_main(member: PartyMember):
        logger.info("Starting thread %s" % threading.current_thread().name)
        comm = member_communicator(
            participant=member,
            world_size=config.common.world_size,
            shared_party_info=shared_party_info,
            recv_timeout=config.member.recv_timeout,
        )
        comm.run()
        logger.info("Finishing thread %s" % threading.current_thread().name)

    def local_arbiter_main():
        logger.info("Starting thread %s" % threading.current_thread().name)
        comm = ArbiteredLocalPartyCommunicator(
            participant=arbiter,
            world_size=config.common.world_size,
            shared_party_info=shared_party_info,
            recv_timeout=config.grpc_arbiter.recv_timeout,
        )
        comm.run()
        logger.info("Finishing thread %s" % threading.current_thread().name)

    with reporting(config):
        run_local_agents(
            master=master,
            members=members,
            target_master_func=local_master_main,
            target_member_func=local_member_main,
            arbiter=arbiter,
            target_arbiter_func=local_arbiter_main,
        )
