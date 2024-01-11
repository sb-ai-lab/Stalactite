import logging
import os
import subprocess
from typing import Any, Optional

from docker import APIClient
from docker.errors import APIError

from stalactite.configs import VFLConfig

BASE_CONTAINER_LABEL = 'grpc-experiment'
KEY_CONTAINER_LABEL = 'container-g'
BASE_MASTER_CONTAINER_NAME = 'master-agent-vfl'  # Do not change this value
BASE_MEMBER_CONTAINER_NAME = 'member-agent-vfl'
BASE_IMAGE_FILE = 'grpc-base.dockerfile'
BASE_IMAGE_TAG = 'grpc-base:latest'
PREREQUISITES_NETWORK = 'prerequisites_vfl-network'  # Do not change this value


def get_env_vars(config: VFLConfig) -> dict:
    env_vars = os.environ.copy()
    env_vars['MLFLOW_PORT'] = config.prerequisites.mlflow_port
    env_vars['PROMETHEUS_PORT'] = config.prerequisites.prometheus_port
    env_vars['GRAFANA_PORT'] = config.prerequisites.grafana_port
    env_vars['DOCKER_COMPOSE_PATH'] = config.docker.docker_compose_path

    return env_vars


def is_test_environment() -> bool:
    is_test = int(os.environ.get('STALACTITE_TEST_MODE', '0'))
    return bool(is_test)


def validate_int(value: Any, logger: logging.Logger = logging.getLogger('__main__')):
    try:
        value = int(value)
    except ValueError:
        logger.warning('Invalid argument. Must be positive int or `all`. Defaulting to `10`.')
        value = 10
    return value


def get_logs(
        agent_id: str,
        tail: str = 'all',
        follow: bool = False,
        docker_client: APIClient = APIClient(base_url='unix://var/run/docker.sock'),
        logger: logging.Logger = logging.getLogger('__main__'),
):
    if tail != 'all':
        tail = validate_int(tail)
    try:
        logs_gen = docker_client.logs(container=agent_id, follow=follow, stream=True, tail=tail)
        for log in logs_gen:
            print(log.decode().strip())
    except APIError as exc:
        logger.error('Error retrieving container logs', exc_info=exc)
        raise


def run_subprocess_command(
        command: str,
        logger_err_info: str,
        logger: logging.Logger = logging.getLogger('__main__'),
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
        docker_client: APIClient = APIClient(base_url='unix://var/run/docker.sock'),
):
    try:
        if agent_id is None:
            containers = docker_client.containers(all=True, filters={'label': containers_label})
        else:
            containers = docker_client.containers(all=True, filters={'id': agent_id})
        for container in containers:
            logger.info(f'Container id: {container["Id"]}. Status: {container["Status"]}')
    except APIError as exc:
        logger.error('Error checking container status', exc_info=exc)
        raise
