import os
import logging
import threading
import random
from threading import Thread
from typing import List, Optional

import mlflow
import datasets
import torch

from stalactite.communications.local import LocalMasterPartyCommunicator, LocalMemberPartyCommunicator
from stalactite.base import PartyMember
from stalactite.data_preprocessors import TabularPreprocessor
from stalactite.configs import VFLConfig
from stalactite.party_master_impl import PartyMasterImplResNetSplitNN
from stalactite.party_member_impl import PartyMemberImpl
from examples.utils.prepare_sbol_smm import load_data as load_sbol_smm

logging.basicConfig(level=logging.DEBUG)
logging.getLogger("urllib3").setLevel(logging.CRITICAL)
logger = logging.getLogger(__name__)


def load_processors(config_path: str):

    config = VFLConfig.load_and_validate(config_path)

    if len(os.listdir(config.data.host_path_data_dir)) == 0:
        load_sbol_smm(config.data.host_path_data_dir, parts_num=2)

    dataset = {}
    for m in range(config.common.world_size):
        dataset[m] = datasets.load_from_disk(
            os.path.join(f"{config.data.host_path_data_dir}/part_{m}")
        )

    processors = [
        TabularPreprocessor(dataset=dataset[i], member_id=i, params=config) for i, v in dataset.items()
    ]

    return processors


def run(config_path: Optional[str] = None):
    torch.manual_seed(22)
    config = VFLConfig.load_and_validate(config_path)
    processors = load_processors(config_path)

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
        "model_type": "vfl"

    }

    if config.master.run_mlflow:
        mlflow.log_params(log_params)

    target_uids = [str(i) for i in range(config.data.dataset_size)]

    shared_party_info = dict()

    master = PartyMasterImplResNetSplitNN(
        uid="master",
        epochs=config.common.epochs,
        report_train_metrics_iteration=config.common.report_train_metrics_iteration,
        report_test_metrics_iteration=config.common.report_test_metrics_iteration,
        processor=processors[0],
        target_uids=target_uids,
        batch_size=config.common.batch_size,
        model_update_dim_size=0,
        run_mlflow=config.master.run_mlflow,
        model_name=config.common.vfl_model_name
    )

    member_ids = [f"member-{member_rank}" for member_rank in range(config.common.world_size)]

    members = [
        PartyMemberImpl(
            uid=member_uid,
            member_record_uids=target_uids,
            model_name=config.common.vfl_model_name,
            processor=processors[member_rank],
            batch_size=config.common.batch_size,
            epochs=config.common.epochs,
            report_train_metrics_iteration=config.common.report_train_metrics_iteration,
            report_test_metrics_iteration=config.common.report_test_metrics_iteration,
            is_consequently=config.common.is_consequently,
            members=member_ids if config.common.is_consequently else None,
        )
        for member_rank, member_uid in enumerate(member_ids)
    ]

    def local_master_main():
        logger.info("Starting thread %s" % threading.current_thread().name)
        comm = LocalMasterPartyCommunicator(
            participant=master,
            world_size=config.common.world_size,
            shared_party_info=shared_party_info
        )
        comm.run()
        logger.info("Finishing thread %s" % threading.current_thread().name)

    def local_member_main(member: PartyMember):
        logger.info("Starting thread %s" % threading.current_thread().name)
        comm = LocalMemberPartyCommunicator(
            participant=member,
            world_size=config.common.world_size,
            shared_party_info=shared_party_info,
            master_id=master.id
        )
        comm.run()
        logger.info("Finishing thread %s" % threading.current_thread().name)

    threads = [
        Thread(name=f"main_{master.id}", daemon=True, target=local_master_main),
        *(
            Thread(
                name=f"main_{member.id}",
                daemon=True,
                target=local_member_main,
                args=(member,)
            )
            for member in members
        )
    ]

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()

    if config.master.run_mlflow:
        mlflow.end_run()


if __name__ == "__main__":
    run(config_path="configs/resnet-sbol-vfl.yml")
