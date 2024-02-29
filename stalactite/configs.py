import logging
import os
import warnings
from pathlib import Path
from typing import Literal, Optional, Union, List

import tenseal as ts
import yaml
from pydantic import BaseModel, Field, field_validator, model_validator


def raise_path_not_exist(path: str):
    """
    Helper function to raise if path does not exist.

    :param path: Path to the file / directory
    """
    if not os.path.exists(path):
        raise FileExistsError(f"Path {path} does not exist")


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
    """Common experimental parameters config."""

    epochs: int = Field(default=3, description="Number of epochs to train a model")
    world_size: int = Field(default=2, description="Number of the VFL member agents (without the master)")
    report_train_metrics_iteration: int = Field(
        default=1,
        description="Number of iteration steps between reporting metrics on train dataset split."
    )
    report_test_metrics_iteration: int = Field(
        default=1,
        description="Number of iteration steps between reporting metrics on test dataset split."
    )
    batch_size: int = Field(default=100, description="Batch size used for training")
    experiment_label: str = Field(
        default="default-experiment",
        description="Experiment name used in prerequisites, if unset, defaults to `default-experiment`",
    )
    reports_export_folder: str = Field(
        default=Path(__file__).parent, description="Folder for exporting tests` and experiments` reports"
    )
    rendezvous_timeout: float = Field(default=3600, description="Initial agents rendezvous timeout in sec")
    vfl_model_name: Literal['linreg', 'logreg', 'logreg_sklearn', 'efficientnet', 'mlp', 'resnet'] = Field(
        default='linreg',
        description='Model type. One of `linreg`, `logreg`, `logreg_sklearn`, `efficientnet`, `mlp`, `resnet`'
    )
    is_consequently: bool = Field(default=False, description='Run linear regression updates in sequential mode')
    use_class_weights: bool = Field(default=False, description='Logistic regression')  # TODO
    learning_rate: float = Field(default=0.01, description='Learning rate')
    momentum: float = Field(default=0, description='Momentum')


class DataConfig(BaseModel):
    """Experimental data parameters config."""

    random_seed: int = Field(default=0,
                             description="Experiment data random seed (including random, numpy, torch)")  # TODO use?
    dataset_size: int = Field(default=1000, description="Number of dataset rows to use")
    host_path_data_dir: str = Field(default='.', description="Path to datasets` directory")
    dataset: Literal['mnist', 'sbol', 'smm'] = Field(
        default='mnist',
        description='Dataset type. One of `mnist`, `sbol`'
    )
    use_smm: bool = Field(default=False)  # TODO use?
    dataset_part_prefix: str = Field(default='part_')  # TODO use?
    train_split: str = Field(default='train_train')
    test_split: str = Field(default='train_val')
    features_data_preprocessors: List[str] = Field(default_factory=list)  # TODO use?
    label_data_preprocessors: List[str] = Field(default_factory=list)  # TODO use?
    features_key: str = Field(default="image_part_")
    label_key: str = Field(default="label")


class PrerequisitesConfig(BaseModel):
    """Prerequisites parameters config."""

    mlflow_host: str = Field(default="0.0.0.0", description="MlFlow host")
    mlflow_port: str = Field(default="5000", description="MlFlow port")
    prometheus_host: str = Field(default="0.0.0.0", description="Prometheus host")
    prometheus_port: str = Field(default="9090", description="Prometheus port")
    grafana_port: str = Field(default="3000", description="Grafana port")
    prometheus_server_port: int = 8765


class GRpcConfig(BaseModel):
    """gRPC base parameters config."""

    host: str = Field(default="0.0.0.0", description="Host of the gRPC server and servicer")
    port: str = Field(default="50051", description="Port of the gRPC server")
    max_message_size: int = Field(
        default=-1, description="Maximum message length that the gRPC channel can send or receive. -1 means unlimited"
    )
    server_threadpool_max_workers: int = Field(default=10, description="Concurrent future number of workers")


class GRpcServerConfig(GRpcConfig):
    """gRPC server and servicer parameters config."""


class GRpcArbiterConfig(GRpcConfig):
    """gRPC arbiter server and servicer parameters config."""

    container_host: str = Field(default="0.0.0.0", description="Host of the container with gRPC arbiter service")
    use_arbiter: bool = Field(default=False, description="Whether to include arbiter for VFL with HE")
    grpc_operations_timeout: float = Field(default=300, description="Timeout of the unary calls to gRPC arbiter server")
    ts_algorithm: Literal["CKKS", "BFV"] = Field(default="CKKS", description="Tenseal scheme to use")
    ts_poly_modulus_degree: int = Field(default=8192, description="Tenseal `poly_modulus_degree` param")
    ts_coeff_mod_bit_sizes: Optional[list[int]] = Field(default=None, description="Tenseal `coeff_mod_bit_sizes` param")
    ts_global_scale_pow: int = Field(
        default=20, description="Tenseal `global_scale` parameter will be calculated as 2 ** ts_global_scale_pow"
    )
    ts_plain_modulus: Optional[int] = Field(
        default=None, description="Tenseal `plain_modulus` param. Should not be passed when the scheme is CKKS."
    )
    ts_generate_galois_keys: bool = Field(
        default=True,
        description="Whether to generate galois keys (galois keys are required to do ciphertext rotations)",
    )
    ts_generate_relin_keys: bool = Field(
        default=True, description="Whether to generate relinearization keys (needed for encrypted multiplications)"
    )
    ts_context_path: Optional[str] = Field(default=None, description="Path to saved Tenseal private context file.")

    @field_validator("ts_algorithm")
    @classmethod
    def validate_ts_algorithm(cls, v: str):
        mapping = {
            "CKKS": ts.SCHEME_TYPE.CKKS,
            "BFV": ts.SCHEME_TYPE.BFV,
        }
        return mapping.get(v, ts.SCHEME_TYPE.CKKS)


class PartyConfig(BaseModel):
    """VFL base parties` parameters config."""

    logging_level: Literal["debug", "info", "warning"] = Field(default="info", description="Logging level")
    recv_timeout: float = Field(default=360., description='Timeout of the recv operation')

    @field_validator("logging_level")
    @classmethod
    def validate_logging_level(cls, v: str):
        level = {
            "info": logging.INFO,
            "warning": logging.WARNING,
            "debug": logging.DEBUG,
        }
        return level.get(v, logging.INFO)


class MasterConfig(PartyConfig):
    """VFL master party`s parameters config."""

    container_host: str = Field(default="0.0.0.0", description="Host of the master container with gRPC server.")
    run_mlflow: bool = Field(default=False, description="Whether to log metrics to MlFlow")
    run_prometheus: bool = Field(default=False, description="Whether to log heartbeats to Prometheus")
    disconnect_idle_client_time: float = Field(
        default=120.0,
        description="Time in seconds to wait after a client`s last heartbeat to consider the client disconnected",
    )
    time_between_idle_connections_checks: float = Field(
        default=3.0, description="Time between checking which clients disconnected"
    )
    master_model_params: dict = Field(default={}, description="Master model parameters")


class MemberConfig(PartyConfig):
    """VFL member parties` parameters config."""

    heartbeat_interval: float = Field(default=2.0, description="Time in seconds to sent heartbeats to master.")
    task_requesting_pings_interval: float = Field(
        default=0.1, description="Interval between new tasks requests from master"
    )  # TODO ?
    sent_task_timout: float = Field(default=3600, description="Timeout of the unary endpoints calls to the gRPC")
    member_model_params: dict = Field(default={}, description="Member model parameters")


class DockerConfig(BaseModel):
    """Docker client parameters config."""

    docker_compose_path: str = Field(
        default=os.path.join(Path(os.path.abspath(__file__)).parent.parent, 'prerequisites'),
        description="Path to the directory containing docker-compose.yml and prerequisites configs"
    )
    docker_compose_command: str = Field(
        default="docker compose", description="Docker compose command to use (`docker compose` | `docker-compose`)"
    )
    use_gpu: bool = Field(
        default=False,
        description='Set to True is your system uses GPU (required for torch dependencies'
    )


class VFLConfig(BaseModel):
    """Experimental parameters general config."""

    common: CommonConfig = Field(default_factory=CommonConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    prerequisites: PrerequisitesConfig = Field(default_factory=PrerequisitesConfig)
    grpc_server: GRpcServerConfig = Field(default_factory=GRpcServerConfig)
    grpc_arbiter: GRpcArbiterConfig = Field(default_factory=GRpcArbiterConfig)
    master: MasterConfig = Field(default_factory=MasterConfig)
    member: MemberConfig = Field(default_factory=MemberConfig)
    docker: DockerConfig = Field(default_factory=DockerConfig)

    config_dir_path: Union[str, Path] = Field(default=Path(__file__).parent.parent)

    @model_validator(mode="after")
    def validate_disconnection_time(self) -> "VFLConfig":
        if self.master.disconnect_idle_client_time - self.member.heartbeat_interval < 2.0:
            raise ValueError(
                f"Heartbeat interval on member (`member.heartbeat_interval`) must be smaller than "
                f"IDLE client`s disconnection time on master (`master.disconnect_idle_client_time`) "
                f"at least by 2 sec.\nCurrent values are {self.member.heartbeat_interval}, "
                f"{self.master.disconnect_idle_client_time}, respectively."
            )

        if self.grpc_arbiter.use_arbiter:
            if (
                    f"{self.grpc_arbiter.host}:{self.grpc_arbiter.port}"
                    == f"{self.grpc_server.host}:{self.grpc_server.port}"
            ):
                raise ValueError(
                    f"Arbiter port {self.grpc_arbiter.port} is the same to "
                    f"gRPC master server port {self.grpc_server.port}"
                )

        return self

    @model_validator(mode="after")
    def set_fields(self) -> "VFLConfig":
        if not self.master.run_prometheus:
            warnings.warn("Reporting to Prometheus is disabled.", UserWarning)
        if not self.master.run_mlflow:
            warnings.warn("Reporting to MlFlow is disabled.", UserWarning)

        docker_path = self.docker.docker_compose_path
        if os.path.exists(docker_path):
            self.docker.docker_compose_path = os.path.abspath(docker_path)
        else:
            self.docker.docker_compose_path = os.path.normpath(os.path.join(self.config_dir_path, docker_path))

        if os.path.isabs(self.data.host_path_data_dir):
            data_dir = self.data.host_path_data_dir
        else:
            data_dir = os.path.normpath(os.path.join(self.config_dir_path, self.data.host_path_data_dir))
            raise_path_not_exist(data_dir)
        self.data.host_path_data_dir = data_dir

        reports_dir = self.common.reports_export_folder
        if not os.path.isabs(reports_dir):
            reports_dir = os.path.normpath(os.path.join(self.config_dir_path, reports_dir))
        os.makedirs(reports_dir, exist_ok=True)
        return self


    @classmethod
    def load_and_validate(cls, config_path: str):
        """Load YAML configuration and validate params with the VFLConfig model."""
        model = load_yaml_config(config_path)
        model['config_dir_path'] = Path(os.path.abspath(config_path)).parent
        return cls.model_validate(model)
