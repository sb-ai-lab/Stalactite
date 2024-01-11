from datetime import datetime, timezone
import logging
import requests
import time

from click.testing import CliRunner
import mlflow
import pytest

from stalactite.main import cli
from stalactite.configs import VFLConfig
from stalactite.utils_main import (
    BASE_CONTAINER_LABEL,
    KEY_CONTAINER_LABEL,
    BASE_MASTER_CONTAINER_NAME,
)


class TestLocalGroupStart:
    def test_local_start_fail(self):
        runner = CliRunner(env={'STALACTITE_TEST_MODE': '1'})
        result = runner.invoke(cli, ['local', '--multi-process', 'start'])
        assert result.exception

    def test_local_start_raises(self, test_config_path):
        runner = CliRunner(env={'STALACTITE_TEST_MODE': '1'})
        result = runner.invoke(
            cli, ['local', '--multi-process', '--single-process', 'start', '--config-path', test_config_path]
        )
        assert 'Either `--single-process` or `--multi-process` flag can be set.' == str(result.exception)

    def test_local_start(self, caplog, test_config_path):
        caplog.set_level(logging.INFO)
        runner = CliRunner(env={'STALACTITE_TEST_MODE': '1'})
        result = runner.invoke(cli, ['local', '--multi-process', 'start', '--config-path', test_config_path])
        assert not result.exception
        assert result.output == 'Stalactite module API\nMultiple-process single-node mode\n'
        master_id_msg = 'Master container id:'
        messages = [
            'Starting multi-process single-node experiment',
            'Building image of the agent',
            'Starting gRPC master container',
            master_id_msg,
            'Starting gRPC member-0 container',
            'Member 0 container id:'
        ]
        for message, log_message in zip(messages, caplog.messages):
            if 'Error while agents containers launch' in log_message:
                raise ValueError('Stop previous experiments` containers')
            assert message in log_message

    def test_raises_created_container(self, test_config_path):
        runner = CliRunner(env={'STALACTITE_TEST_MODE': '1'})
        result = runner.invoke(cli, ['local', '--multi-process', 'start', '--config-path', test_config_path])
        assert result.exception

    @pytest.mark.timeout(300)
    def test_master_container_running(self, docker_client):
        status = ''
        while 'Exited' not in status:
            master_container = docker_client.containers(
                all=True,
                filters={'name': f'{BASE_MASTER_CONTAINER_NAME}-test', 'status': ['running', 'exited']}
            )
            assert len(master_container) == 1
            status = master_container[0]["Status"]
            time.sleep(3)

    def test_local_results(self, test_config_path, docker_client):
        config = VFLConfig.load_and_validate(test_config_path)
        response_prometheus = requests.get(
            f'http://{config.prerequisites.prometheus_host}:{config.prerequisites.prometheus_port}/api/v1/query',
            params={
                'query': 'number_of_connected_agents{' f'experiment_label="{config.common.experiment_label}"' '}[5m]'
            }
        )
        body_prometheus = response_prometheus.json()
        mlflow.set_tracking_uri(f"http://{config.prerequisites.mlflow_host}:{config.prerequisites.mlflow_port}")
        response_mlflow = mlflow.search_runs(experiment_names=[config.common.experiment_label])

        assert response_prometheus.status_code == 200
        assert body_prometheus['status'] == 'success'
        assert len(body_prometheus['data']['result']) > 0
        assert response_mlflow.shape[0] > 0
        assert len({'metrics.test_mae', 'metrics.test_acc', 'metrics.train_mae', 'metrics.train_acc'}
                   .intersection(set(response_mlflow.columns))) == 4
        assert (datetime.now(timezone.utc) - response_mlflow.iloc[0]['end_time'].to_pydatetime()).seconds < 240


class TestLocalGroupStop:
    def test_stop_experiment_containers(self, test_config_path, caplog):
        config = VFLConfig.load_and_validate(test_config_path)
        caplog.set_level(logging.INFO)
        runner = CliRunner(env={'STALACTITE_TEST_MODE': '1'})
        result = runner.invoke(cli, ['local', '--multi-process', 'stop', '--leave-containers'])
        assert not result.exception
        assert result.output == 'Stalactite module API\nMultiple-process single-node mode\n'
        assert len(caplog.messages) == 2 + config.common.world_size
        for message in caplog.messages:
            assert 'Stopping' in message

    def test_stop_remove_experiment_containers(self, test_config_path, caplog):
        config = VFLConfig.load_and_validate(test_config_path)
        caplog.set_level(logging.INFO)
        runner = CliRunner(env={'STALACTITE_TEST_MODE': '1'})
        result = runner.invoke(cli, ['local', '--multi-process', 'stop'])
        assert not result.exception
        assert result.output == 'Stalactite module API\nMultiple-process single-node mode\n'
        assert len(caplog.messages) == 1 + (config.common.world_size + 1) * 2
        for message in caplog.messages:
            assert 'Stopping' in message or 'Removing' in message

    def test_removed_integration_containers(self, docker_client):
        test_containers = docker_client.containers(
            all=True,
            filters={'label': f'{KEY_CONTAINER_LABEL}={BASE_CONTAINER_LABEL + "test"}'}
        )
        assert len(test_containers) == 0
