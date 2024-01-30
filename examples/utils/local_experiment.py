import os
import logging
import pickle
from pathlib import Path
import uuid
import random
import threading
from threading import Thread
from typing import List, Optional

import torch
import mlflow
import numpy as np
import datasets
import scipy as sp
from sklearn.metrics import mean_absolute_error,  roc_auc_score, precision_recall_curve, auc
from datasets import DatasetDict
from sklearn.linear_model import LogisticRegression as LogRegSklearn

from stalactite.models.linreg_batch import LinearRegressionBatch
from stalactite.models.logreg_batch import LogisticRegressionBatch
from stalactite.party_member_impl import PartyMemberImpl
from stalactite.data_loader import load, init, DataPreprocessor
from stalactite.data_preprocessors import ImagePreprocessor, TabularPreprocessor
from stalactite.batching import ListBatcher
from stalactite.metrics import ComputeAccuracy_numpy
from stalactite.party_master_impl import PartyMasterImpl, PartyMasterImplConsequently, PartyMasterImplLogreg
from stalactite.communications.local import LocalMasterPartyCommunicator, LocalMemberPartyCommunicator
from stalactite.base import PartyMember
from stalactite.configs import VFLConfig

logging.basicConfig(level=logging.DEBUG)
logging.getLogger("urllib3").setLevel(logging.CRITICAL)
logger = logging.getLogger(__name__)


def compute_class_weights(classes_idx, y_train) -> torch.Tensor:
    pos_weights_list = []
    for i, c_idx in enumerate(classes_idx):
        unique, counts = np.unique(y_train[:, i], return_counts=True)
        if unique.shape[0] < 2:
            raise ValueError(f"class {c_idx} has no label 1")
        pos_weights_list.append(counts[0] / counts[1])
    return torch.tensor(pos_weights_list)


def compute_class_distribution(classes_idx: list, y: torch.Tensor, name: str) -> None:
    logger.info(f"{name} distribution")
    for i, c_idx in enumerate(classes_idx):
        unique, counts = np.unique(y[:, i], return_counts=True)
        logger.info(f"for class: {c_idx}")
        logger.info(np.asarray((unique, counts)).T)
        if unique.shape[0] < 2:
            raise ValueError(f"class {c_idx} has no label 1")


def load_parameters(config_path: str):
    # BASE_PATH = Path(__file__).parent.parent.parent

    config = VFLConfig.load_and_validate(config_path)
    sample_size = int(os.environ.get("SAMPLE_SIZE", 10000))

    # models input dims for 1, 2, 3 and 5 members
    if config.data.dataset.lower() == "mnist":
        # todo: hide input_dim in preprocessor
        input_dims_list = [[619], [392, 392], [204, 250, 165], [], [108, 146, 150, 147, 68]]
        params = init(config_path=os.path.abspath(config_path))
    elif config.data.dataset.lower() == "sbol":
        smm = "smm_" if config.data.use_smm else ""
        dim = 1356 if smm == "smm_" else 1345
        input_dims_list = [[0], [1345, 11]]

    if config.data.dataset.lower() == "mnist":
        dataset, _ = load(params)
        processors = [
            ImagePreprocessor(dataset=dataset[i], member_id=i, data_params=params[i].data) for i, v in dataset.items()
        ]

    elif config.data.dataset.lower() == "sbol":

        dataset = {}

        for m in range(config.common.world_size):
            dataset[m] = datasets.load_from_disk(
                os.path.join(f"{config.data.host_path_data_dir}/part_{m}")
            )
        processors = [
            TabularPreprocessor(dataset=dataset[i], member_id=i, data_params=config.data) for i, v in dataset.items()
        ]
        input_dims_list = [[0], [1345, 11]]

    else:
        raise ValueError(f"Unknown dataset: {config.data.dataset}, choose one from ['mnist', 'multilabel']")

    return input_dims_list, config.data, processors

# parametrized file __main__ run() с хардкод

# TODO
# запушить ветку заребейзив ветку Димы на себя -> pr в мейн
# loop (dima)
# local distributed
# distributed
def run(config_path: Optional[str] = None):
    if config_path is None:
        config_path = os.environ.get(
            'SINGLE_MEMBER_CONFIG_PATH',
            os.path.join(Path(__file__).parent.parent.parent, 'configs/linreg-mnist-local.yml')
        )
    config = VFLConfig.load_and_validate(config_path)

    if config.master.run_mlflow:
        mlflow.set_tracking_uri(f"http://{config.prerequisites.mlflow_host}:{config.prerequisites.mlflow_port}")
        mlflow.set_experiment(config.common.experiment_label)
        mlflow.start_run()

    model_name = config.common.vfl_model_name


    if 'logreg' in model_name:
        # todo: hide it somehow
        classes_idx = [x for x in range(19)]
    else:
        classes_idx = list()
    n_labels = len(classes_idx)

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
        "n_labels": n_labels,

    }

    if config.master.run_mlflow:
        mlflow.log_params(log_params)

    input_dims_list, params, processors = load_parameters(config_path)

    # todo: hide this in preprocessor
    num_dataset_records = [200 + random.randint(100, 1000) for _ in range(config.common.world_size)]
    shared_record_uids = [str(i) for i in range(config.data.dataset_size)]
    target_uids = shared_record_uids

    # todo: add assigning class weights to preprocessor

    #     class_weights = compute_class_weights(classes_idx, targets) if config.common.use_class_weights else None
    #     if config.master.run_mlflow:
    #         mlflow.log_param("class_weights", class_weights)
    #     compute_class_distribution(classes_idx=classes_idx, y=test_targets, name="test")

    members_datasets_uids = [
        [*shared_record_uids, *(str(uuid.uuid4()) for _ in range(num_records - len(shared_record_uids)))]
        for num_records in num_dataset_records
    ]
    shared_party_info = dict()
    if 'logreg' in config.common.vfl_model_name:
        master_class = PartyMasterImplLogreg
    else:
        if config.common.is_consequently:
            master_class = PartyMasterImplConsequently
        else:
            master_class = PartyMasterImpl
    master = master_class(
        uid="master",
        epochs=config.common.epochs,
        report_train_metrics_iteration=config.common.report_train_metrics_iteration,
        report_test_metrics_iteration=config.common.report_test_metrics_iteration,
        processor=processors[0],
        target_uids=target_uids,
        batch_size=config.common.batch_size,
        model_update_dim_size=0,
        run_mlflow=config.master.run_mlflow,
    )

    if 'logreg' in model_name:
        model = lambda member_rank: LogisticRegressionBatch(
            input_dim=input_dims_list[config.common.world_size - 1][member_rank],
            output_dim=n_labels,
            learning_rate=config.common.learning_rate,
            class_weights=None,
            init_weights=0.005
        )
    else:
        model = lambda member_rank: LinearRegressionBatch(
            input_dim=input_dims_list[config.common.world_size - 1][member_rank],
            output_dim=1,
            reg_lambda=0.2
        )

    members = [
        PartyMemberImpl(
            uid=f"member-{member_rank}",
            model_update_dim_size=input_dims_list[config.common.world_size - 1][member_rank],
            member_record_uids=member_uids,
            model=model(member_rank),
            processor=processors[member_rank],
            batch_size=config.common.batch_size,
            epochs=config.common.epochs,
            report_train_metrics_iteration=config.common.report_train_metrics_iteration,
            report_test_metrics_iteration=config.common.report_test_metrics_iteration,
        )
        for member_rank, member_uids in enumerate(members_datasets_uids)
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
    run()
