import logging
import os
from pathlib import Path
from typing import Union, Literal

import yaml
from pydantic import BaseModel, Field, field_validator


def raise_path_not_exist(path: str):
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
    if not os.path.exists(yaml_path):
        raise ValueError(f"Configuration file at `config-path` {yaml_path} does not exist")

    with open(yaml_path, "r") as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            raise ValueError(f"Check YAML file at `config-path` {yaml_path}") from exc


class CommonConfig(BaseModel):
    # TODO add docs
    random_seed: int
    epochs: int
    world_size: int
    report_train_metrics_iteration: int = Field(default=1)
    report_test_metrics_iteration: int = Field(default=1)
    batch_size: int = Field(default=10)


class DataConfig(BaseModel):
    random_seed: int = Field(default=0)
    dataset_size: int
    host_path_data_dir: str = Field()


class PrerequisitesConfig(BaseModel):
    run_mlflow: bool = Field(default=False, description='Whether to log metrics to MlFlow')
    mlflow_host: str = Field(default='0.0.0.0')
    mlflow_port: str = Field(default='5000')


class GRpcServerConfig(BaseModel):
    host: str = Field(default='0.0.0.0')
    port: str = Field(default='50051')
    max_message_size: int = Field(
        default=-1,
        description='Maximum message length that the gRPC channel can send or receive. -1 means unlimited.'
    )


class PartyConfig(BaseModel):
    logging_level: Literal['debug', 'info', 'warning'] = Field(default='info')

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
    pass


class MemberConfig(PartyConfig):
    pass


class DockerConfig(BaseModel):
    docker_compose_path: str = Field(default='../prerequisites')
    docker_compose_command: str = Field(default="docker compose")


class VFLConfig(BaseModel):
    common: CommonConfig
    data: DataConfig
    prerequisites: PrerequisitesConfig
    grpc_server: GRpcServerConfig
    master: MasterConfig
    member: MemberConfig
    docker: DockerConfig
