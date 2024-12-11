import logging
import os
import warnings
from pathlib import Path
from typing import Literal, Optional, Union

import yaml
from pydantic import BaseModel, Field, model_validator


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
    use_grpc: bool = Field(default=False, description="Whether to use local or gRPC-based communicators")

    world_size: int = Field(default=2, description="Number of the VFL member agents (without the master)")
    report_train_metrics_iteration: int = Field(
        default=1,
        description="Number of iteration steps between reporting metrics on train dataset split."
    )
    report_test_metrics_iteration: int = Field(
        default=1,
        description="Number of iteration steps between reporting metrics on test dataset split."
    )
    experiment_label: str = Field(
        default="default-experiment",
        description="Experiment name used in prerequisites, if unset, defaults to `default-experiment`",
    )
    reports_export_folder: str = Field(
        default=Path(__file__).parent, description="Folder for exporting tests` and experiments` reports"
    )
    rendezvous_timeout: float = Field(default=3600, description="Initial agents rendezvous timeout in sec")
    seed: int = Field(default=42, description="Initial random seed")
    logging_level: Literal["debug", "info", "warning"] = Field(default="info", description="Logging level")

    @model_validator(mode="after")
    def validate_logging_level(self):
        level = {
            "info": logging.INFO,
            "warning": logging.WARNING,
            "debug": logging.DEBUG,
        }
        self.logging_level = level.get(self.logging_level, logging.INFO)
        return self


class VFLModelConfig(BaseModel):
    epochs: int = Field(default=3, description="Number of epochs to train a model")
    batch_size: int = Field(default=100, description="Batch size used for training")
    eval_batch_size: int = Field(default=100, description="Batch size used for evaluation")
    vfl_model_name: str = Field(
        default='linreg',
        description='Model type. One of `linreg`, `logreg`, `logreg_sklearn`, `efficientnet`, `mlp`, `resnet`'
    )
    is_consequently: bool = Field(default=False, description='Run linear regression updates in sequential mode')
    use_class_weights: bool = Field(default=False, description='Imbalanced classes logistic regression class weights')
    learning_rate: float = Field(default=0.01, description='Learning rate')
    l2_alpha: Optional[float] = Field(default=None, description='Alpha used for L2 regularization')
    momentum: Optional[float] = Field(default=0, description='Optimizer momentum')
    weight_decay: Optional[float] = Field(default=0.01, description='Optimizer weight decay')
    do_train: bool = Field(default=True, description='Whether to run a training loop.')
    do_predict: bool = Field(default=True, description='Whether to run an inference loop.')
    do_save_model: bool = Field(default=True, description='Whether to save the model after training.')
    vfl_model_path: str = Field(
        default='.',
        description="Directory to save the model after the training or load the model for inference"
    )


class DataConfig(BaseModel):
    """Experimental data parameters config."""

    dataset_size: int = Field(default=100, description="Number of dataset rows to use")
    host_path_data_dir: str = Field(default='.', description="Path to datasets` directory")
    dataset: Literal[
        'sbol_master_only_labels',
        'mnist',
        'sbol',
        'sbol_smm',
        'home_credit',
        'home_credit_bureau_pos',
        'avito',
        'avito_texts_images',
    ] = Field(
        default='mnist',
        description='Dataset type. One of `mnist`, `sbol`, `sbol_smm`, `home_credit`,  `home_credit_bureau_pos`,'
                    ' `avito`, `avito_texts_images`'
    )
    dataset_part_prefix: str = Field(default='part_')
    train_split: str = Field(default='train_train')
    test_split: str = Field(default='train_val')
    features_key: str = Field(default="image_part_")
    label_key: str = Field(default="label")
    uids_key: str = Field(default="user_id")
    num_classes: int = Field(default=1)


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

    port: str = Field(default="50051", description="Port of the gRPC server")
    max_message_size: int = Field(
        default=-1, description="Maximum message length that the gRPC channel can send or receive. -1 means unlimited"
    )
    server_threadpool_max_workers: int = Field(default=10, description="Concurrent future number of workers")


class GRpcServerConfig(GRpcConfig):
    """gRPC server and servicer parameters config."""


class PaillierSPParams(BaseModel):
    """ Security protocol parameters if the Paillier is used. """
    he_type: Literal['paillier']
    encryption_precision: float = Field(default=1e-8, description='Precision of the Paillier encryption.')
    encoding_precision: float = Field(default=1e-8, description='Precision of the Paillier encoding.')
    key_length: int = Field(default=2048, description='Length of the Paillier cryptokey.')
    n_threads: int = Field(default=None, description='Number of threads to use for computations')

    @property
    def init_params(self):
        return {
            "encryption_precision": self.encryption_precision,
            "encoding_precision": self.encoding_precision,
            "key_length": self.key_length,
            "n_threads": self.n_threads,
        }


class PartyConfig(BaseModel):
    """VFL base parties` parameters config."""
    logging_level: Literal["debug", "info", "warning"] = Field(default="info", description="Logging level")
    recv_timeout: float = Field(default=360., description='Timeout of the recv operation')
    cuda_visible_devices: str = Field(
        default='all',
        description='CUDA_VISIBLE_DEVICES ids. E.g. "0,2,3" for CUDA to use GPUs with ids 0, 2 and 3 only.'
    )

    @model_validator(mode="after")
    def validate_logging_level(self):
        level = {
            "info": logging.INFO,
            "warning": logging.WARNING,
            "debug": logging.DEBUG,
        }
        self.logging_level = level.get(self.logging_level, logging.INFO)
        return self


class GRpcArbiterConfig(GRpcConfig, PartyConfig):
    """gRPC arbiter server and servicer parameters config."""
    external_host: str = Field(
        default="0.0.0.0",
        description="Host of the node with the container with gRPC arbiter service"
    )
    use_arbiter: bool = Field(default=False, description="Whether to include arbiter for VFL with HE")
    grpc_operations_timeout: float = Field(default=300, description="Timeout of the unary calls to gRPC arbiter server")
    security_protocol_params: Optional[PaillierSPParams] = Field(default=None)


class MasterConfig(PartyConfig):
    """VFL master party`s parameters config."""

    external_host: str = Field(
        default="0.0.0.0",
        description="Host of the node with the master container with gRPC server."
    )
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
    vfl_model: VFLModelConfig = Field(default_factory=VFLModelConfig)
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

        if self.grpc_arbiter.use_arbiter and self.common.use_grpc:
            if str(self.grpc_arbiter.port) == str(self.grpc_server.port):
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

        if not os.path.isabs(self.vfl_model.vfl_model_path):
            self.vfl_model.vfl_model_path = os.path.normpath(
                os.path.join(self.config_dir_path, self.vfl_model.vfl_model_path)
            )
        os.makedirs(self.vfl_model.vfl_model_path, exist_ok=True)

        return self

    @classmethod
    def load_and_validate(cls, config_path: str):
        """Load YAML configuration and validate params with the VFLConfig model."""
        model = load_yaml_config(config_path)
        model['config_dir_path'] = Path(os.path.abspath(config_path)).parent
        return cls.model_validate(model)
