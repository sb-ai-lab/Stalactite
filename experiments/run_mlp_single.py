import os
import logging
from typing import Optional

import mlflow
import datasets
import torch


from stalactite.data_preprocessors import TabularPreprocessor
from stalactite.configs import VFLConfig
from stalactite.party_single_impl import PartySingleMLP, PartySingleMLPSplitNN
from examples.utils.prepare_sbol_smm import load_data as load_sbol_smm

logging.basicConfig(level=logging.DEBUG)
logging.getLogger("urllib3").setLevel(logging.CRITICAL)
logger = logging.getLogger(__name__)


def load_processors(config_path: str):

    config = VFLConfig.load_and_validate(config_path)

    if len(os.listdir(config.data.host_path_data_dir)) == 0:
        load_sbol_smm(config.data.host_path_data_dir, parts_num=1)

    dataset = {0: datasets.load_from_disk(
        os.path.join(f"{config.data.host_path_data_dir}/part_{0}")
    )}

    processors = [
        TabularPreprocessor(dataset=dataset[i], member_id=i, params=config) for i, v in dataset.items()
    ]

    return processors


def run(config_path: Optional[str] = None):
    torch.manual_seed(22)
    config = VFLConfig.load_and_validate(config_path)

    if config.master.run_mlflow:
        mlflow.set_tracking_uri(f"http://{config.prerequisites.mlflow_host}:{config.prerequisites.mlflow_port}")
        mlflow.set_experiment(config.common.experiment_label)
        mlflow.start_run()

    model_name = config.common.vfl_model_name

    log_params = {
        "ds_size": config.data.dataset_size,
        "batch_size": config.common.batch_size,
        "epochs": config.common.epochs,
        "mode": "vfl",
        "members_count": config.common.world_size,
        "exp_uid": config.common.experiment_label,
        "is_consequently": config.common.is_consequently,
        "model_name": model_name,
        "learning_rate": config.common.learning_rate,
        "dataset": config.data.dataset,

    }

    if config.master.run_mlflow:
        mlflow.log_params(log_params)

    processors = load_processors(config_path)
    target_uids = [str(i) for i in range(config.data.dataset_size)]
    divided = True

    if divided:
        party = PartySingleMLPSplitNN(
            processor=processors[0],
            batch_size=config.common.batch_size,
            epochs=config.common.epochs,
            report_train_metrics_iteration=config.common.report_train_metrics_iteration,
            report_test_metrics_iteration=config.common.report_test_metrics_iteration,
            use_mlflow=config.master.run_mlflow,
            target_uids=target_uids
        )

    else:
        party = PartySingleMLP(
            processor=processors[0],
            batch_size=config.common.batch_size,
            epochs=config.common.epochs,
            report_train_metrics_iteration=config.common.report_train_metrics_iteration,
            report_test_metrics_iteration=config.common.report_test_metrics_iteration,
            use_mlflow=config.master.run_mlflow,
            target_uids=target_uids
        )

    party.run()

    if config.master.run_mlflow:
        mlflow.end_run()


if __name__ == "__main__":
    run(config_path="configs/mlp-sbol-single.yml")
