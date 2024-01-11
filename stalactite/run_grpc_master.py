from dataclasses import dataclass
import random
import os
import pickle

import click
import mlflow
import numpy as np
import torch

from stalactite.communications import GRpcMasterPartyCommunicator
from stalactite.configs import VFLConfig
from stalactite.party_master_impl import PartyMasterImpl
from stalactite.base import DataTensor


@dataclass
class MasterData:
    targets: DataTensor
    target_uids: list[str]
    test_targets: DataTensor


def prepare_data(config: VFLConfig) -> MasterData:
    random.seed(config.data.random_seed)
    np.random.seed(config.data.random_seed)
    torch.manual_seed(config.data.random_seed)
    torch.cuda.manual_seed_all(config.data.random_seed)
    torch.backends.cudnn.deterministic = True

    shared_uids_count = config.data.dataset_size
    shared_record_uids = [str(i) for i in range(shared_uids_count)]
    target_uids = shared_record_uids
    with open(
            os.path.join(
                os.path.expanduser(config.data.host_path_data_dir),
                f"datasets_{config.common.world_size}_members.pkl"
            ),
            'rb'
    ) as f:
        datasets_list = pickle.load(f)['data']

    targets = datasets_list[0]["train_train"]["label"][:config.data.dataset_size]
    test_targets = datasets_list[0]["train_val"]["label"]
    return MasterData(targets=targets, target_uids=target_uids, test_targets=test_targets)


def get_party_master(
        master_data: MasterData,
        epochs: int = 1,
        report_train_metrics_iteration: int = 1,
        report_test_metrics_iteration: int = 1,
        batch_size: int = 10,
        run_mlflow: bool = False,
):
    return PartyMasterImpl(
        uid="master",
        epochs=epochs,
        report_train_metrics_iteration=report_train_metrics_iteration,
        report_test_metrics_iteration=report_test_metrics_iteration,
        target=master_data.targets,
        test_target=master_data.test_targets,
        target_uids=master_data.target_uids,
        batch_size=batch_size,
        model_update_dim_size=0,
        run_mlflow=run_mlflow,
    )

@click.command()
@click.option('--config-path', type=str, default='../configs/config.yml')
def main(config_path):
    config = VFLConfig.load_and_validate(config_path)
    if config.master.run_mlflow:
        mlflow.set_tracking_uri(f"http://{config.prerequisites.mlflow_host}:{config.prerequisites.mlflow_port}")
        mlflow.set_experiment(config.common.experiment_label)
        mlflow.start_run()

    data = prepare_data(config)
    party_master = get_party_master(
        master_data=data,
        epochs=config.common.epochs,
        report_train_metrics_iteration=config.common.report_train_metrics_iteration,
        report_test_metrics_iteration=config.common.report_test_metrics_iteration,
        batch_size=config.common.batch_size,
        run_mlflow=config.master.run_mlflow
    )
    comm = GRpcMasterPartyCommunicator(
        participant=party_master,
        world_size=config.common.world_size,
        port=config.grpc_server.port,
        host=config.grpc_server.host,
        max_message_size=config.grpc_server.max_message_size,
        logging_level=config.master.logging_level,
        prometheus_server_port=config.prerequisites.prometheus_server_port,
        run_prometheus=config.master.run_prometheus,
        experiment_label=config.common.experiment_label,
    )
    comm.run()
    if config.master.run_mlflow:
        mlflow.end_run()


if __name__ == '__main__':
    main()
