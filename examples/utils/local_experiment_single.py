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
from stalactite.party_single_impl import PartySingleLinreg#, PartySingleLogreg

logging.basicConfig(level=logging.DEBUG)
logging.getLogger("urllib3").setLevel(logging.CRITICAL)
logger = logging.getLogger(__name__)


def load_parameters(config_path: str):
    # BASE_PATH = Path(__file__).parent.parent.parent

    config = VFLConfig.load_and_validate(config_path)

    dataset = {0: datasets.load_from_disk(
        os.path.join(f"{config.data.host_path_data_dir}/part_{0}")
    )}

    if config.data.dataset.lower() == "mnist":

        processors = [
            ImagePreprocessor(dataset=dataset[i], member_id=i, params=config) for i, v in dataset.items()
        ]

    elif config.data.dataset.lower() == "sbol":

        processors = [
            TabularPreprocessor(dataset=dataset[i], member_id=i, params=config) for i, v in dataset.items()
        ]

    else:
        raise ValueError(f"Unknown dataset: {config.data.dataset}, choose one from ['mnist', 'multilabel']")

    return config.data, processors


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

    params, processors = load_parameters(config_path)

    # uid = member_uid,
    # member_record_uids = target_uids,
    # model_name = config.common.vfl_model_name,
    # processor = processors[member_rank],
    # batch_size = config.common.batch_size,
    # epochs = config.common.epochs,
    # report_train_metrics_iteration = config.common.report_train_metrics_iteration,
    # report_test_metrics_iteration = config.common.report_test_metrics_iteration,
    # is_consequently = config.common.is_consequently,
    # members = member_ids if config.common.is_consequently else None,

    if model_name == "linreg":
        party = PartySingleLinreg(
            processor=processors[0],
            batch_size=config.common.batch_size,
            epochs=config.common.epochs,
            report_train_metrics_iteration=config.common.report_train_metrics_iteration,
            report_test_metrics_iteration=config.common.report_test_metrics_iteration,
            use_mlflow=config.master.run_mlflow
        )

    # dataset = datasets_list,
    # dataset_size = config.data.dataset_size,
    # epochs = epochs,
    # batch_size = batch_size,
    # features_key = features_key,
    # labels_key = labels_key,
    # output_dim = output_dim,
    # use_mlflow = config.master.run_mlflow,
    # test_inner_users = None,
    # report_train_metrics_iteration = config.common.report_train_metrics_iteration,
    # report_test_metrics_iteration = config.common.report_test_metrics_iteration,
    # is_multilabel = is_multilabel,
    # input_dims = [619],
    # learning_rate = learning_rate,


    elif model_name == "logreg":
        party = PartySingleLogreg(
            dataset=datasets_list,
            dataset_size=config.data.dataset_size,
            epochs=epochs,
            batch_size=batch_size,
            features_key=features_key,
            labels_key=labels_key,
            output_dim=output_dim,
            use_mlflow=config.master.run_mlflow,
            test_inner_users=None,
            report_train_metrics_iteration=config.common.report_train_metrics_iteration,
            report_test_metrics_iteration=config.common.report_test_metrics_iteration,
            is_multilabel=is_multilabel,
            input_dims=[619],
            learning_rate=learning_rate,
        )
    else:
        raise ValueError(f"unknown model name: {model_name}")

    party.run()

    if config.master.run_mlflow:
        mlflow.end_run()


if __name__ == "__main__":
    run()
