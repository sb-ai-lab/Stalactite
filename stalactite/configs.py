import logging
import os
from pathlib import Path
from typing import Union, Literal
import warnings

from pydantic import BaseModel, Field, field_validator, model_validator
import yaml


def raise_path_not_exist(path: str):
    """
    Helper function to raise if path does not exist.

    :param path: Path to the file / directory
    """
    if not os.path.exists(path):
        raise FileExistsError(f'Path {path} does not exist')


class AttrDict(dict):
    # TODO replace with pydantic
    """ Attribute Dictionary"""

    def __setitem__(self, key, value):
        if isinstance(value, dict):  # change dict to AttrDict
            value = AttrDict(value)
        super(AttrDict, self).__setitem__(key, value)

    def __getitem__(self, key):
        value = self.get(key)
        # this part is needed when we initialized the AttrDict datastructure form recursive dict.
        if not isinstance(value, AttrDict):
            # dynamically change the type of value from dict to AttrDict
            if isinstance(value, dict):
                value = AttrDict(value)
                self.__setitem__(key, value)
        return self.get(key)

    def __setattr__(self, key, value):
        # import pdb; pdb.set_trace()
        self.__setitem__(key, value)

    def __getattr__(self, key):
        # import pdb; pdb.set_trace()
        return self.__getitem__(key)

    __delattr__ = dict.__delitem__

    def __add__(self, other):
        res = AttrDict({**self, **other})
        return res


def load_yaml_config(yaml_path: Union[str, Path]) -> dict:
    """
    Load YAML file from yaml_path.

    :param yaml_path: Path of the YAML file to load
    """
    if not os.path.exists(yaml_path):
        raise ValueError(f"Configuration file at `config-path` {yaml_path} does not exist")

    with open(yaml_path, "r") as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            raise ValueError(f"Check YAML file at `config-path` {yaml_path}") from exc


class CommonConfig(BaseModel):
    """ Common experimental parameters config. """
    epochs: int = Field(description='Number of epochs to train a model')
    world_size: int = Field(description='Number of the VFL member agents (without the master)')
    report_train_metrics_iteration: int = Field(default=1)
    report_test_metrics_iteration: int = Field(default=1)
    batch_size: int = Field(default=10, description='Batch size used for training')
    experiment_label: str = Field(
        default='default-experiment',
        description='Experiment name used in prerequisites, if unset, defaults to `default-experiment`'
    )
    reports_export_folder: str = Field(
        default=Path(__file__).parent,
        description='Folder for exporting tests` and experiments` reports'
    )

    @field_validator('reports_export_folder')
    @classmethod
    def check_if_exists_or_create(cls, v: str):
        os.makedirs(v, exist_ok=True)
        return v


class DataConfig(BaseModel):
    """ Experimental data parameters config. """
    random_seed: int = Field(default=0, description='Experiment data random seed (including random, numpy, torch)')
    dataset_size: int = Field(description='Number of dataset rows to use')
    host_path_data_dir: str = Field(description='Path to datasets` directory')


class PrerequisitesConfig(BaseModel):
    """ Prerequisites parameters config. """
    mlflow_host: str = Field(default='0.0.0.0', description='MlFlow host')
    mlflow_port: str = Field(default='5000', description='MlFlow port')
    prometheus_host: str = Field(default='0.0.0.0', description='Prometheus host')
    prometheus_port: str = Field(default='9090', description='Prometheus port')
    grafana_port: str = Field(default='3000', description='Grafana port')
    prometheus_server_port: int = 8765


class GRpcServerConfig(BaseModel):
    """ gRPC server and servicer parameters config. """
    host: str = Field(default='0.0.0.0', description='Host of the gRPC server and servicer')
    port: str = Field(default='50051', description='Port of the gRPC server')
    max_message_size: int = Field(
        default=-1,
        description='Maximum message length that the gRPC channel can send or receive. -1 means unlimited.'
    )


class PartyConfig(BaseModel):
    """ VFL base parties` parameters config. """
    logging_level: Literal['debug', 'info', 'warning'] = Field(default='info', description='Logging level')

    @field_validator('logging_level')
    @classmethod
    def validate_logging_level(cls, v: str):
        level = {
            'info': logging.INFO,
            'warning': logging.WARNING,
            'debug': logging.DEBUG,
        }
        return level.get(v, logging.INFO)


class MasterConfig(PartyConfig):
    """ VFL master party`s parameters config. """
    run_mlflow: bool = Field(default=False, description='Whether to log metrics to MlFlow')
    run_prometheus: bool = Field(default=False, description='Whether to log heartbeats to Prometheus')
    disconnect_idle_client_time: float = Field(
        default=120.,
        description='Time in seconds to wait after a client`s last heartbeat to consider the client disconnected'
    )


class MemberConfig(PartyConfig):
    """ VFL member parties` parameters config. """
    heartbeat_interval: float = Field(default=2., description='Time in seconds to sent heartbeats to master.')


class DockerConfig(BaseModel):
    """ Docker client parameters config. """
    docker_compose_path: str = Field(
        default='../prerequisites',
        description='Path to the directory containing docker-compose.yml'
    )
    docker_compose_command: str = Field(
        default="docker compose",
        description='Docker compose command to use (`docker compose` | `docker-compose`)')

    @field_validator('docker_compose_path')
    @classmethod
    def validate_docker_folder(cls, v: str):
        if os.path.exists(v):
            return os.path.abspath(v)
        else:
            return v


class VFLConfig(BaseModel):
    """ Experimental parameters general config. """
    common: CommonConfig
    data: DataConfig
    prerequisites: PrerequisitesConfig
    grpc_server: GRpcServerConfig
    master: MasterConfig
    member: MemberConfig
    docker: DockerConfig

    @model_validator(mode='after')
    def validate_disconnection_time(self) -> 'VFLConfig':
        if self.master.disconnect_idle_client_time - self.member.heartbeat_interval < 2.:
            raise ValueError(
                f'Heartbeat interval on member (`member.heartbeat_interval`) must be smaller than '
                f'IDLE client`s disconnection time on master (`master.disconnect_idle_client_time`) '
                f'at least by 2 sec.\nCurrent values are {self.member.heartbeat_interval}, '
                f'{self.master.disconnect_idle_client_time}, respectively.')
        return self

    @model_validator(mode='after')
    def info_set_reports(self) -> 'VFLConfig':
        if not self.master.run_prometheus:
            warnings.warn('Reporting to Prometheus is disabled.', UserWarning)
        if not self.master.run_mlflow:
            warnings.warn('Reporting to MlFlow is disabled.', UserWarning)
        return self

    @classmethod
    def load_and_validate(cls, config_path: str):
        return cls.model_validate(load_yaml_config(config_path))
