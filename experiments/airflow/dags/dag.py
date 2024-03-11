import os
import sys
import pickle

from typing import List
import sys
import logging

import pendulum
from datetime import timedelta

import threading
from typing import Optional
import datasets

from stalactite.ml import (
    HonestPartyMasterLinRegConsequently,
    HonestPartyMasterLinReg,
    HonestPartyMemberLogReg,
    HonestPartyMemberLinReg,
    HonestPartyMasterLogReg,
    HonestPartyMemberResNet,
    HonestPartyMasterResNetSplitNN,
    HonestPartyMemberEfficientNet,
    HonestPartyMasterEfficientNetSplitNN,
    HonestPartyMasterMLPSplitNN,
    HonestPartyMemberMLP
)
from stalactite.data_preprocessors import ImagePreprocessor, TabularPreprocessor, ImagePreprocessorEff
from stalactite.communications.local import LocalMasterPartyCommunicator, LocalMemberPartyCommunicator
from stalactite.base import PartyMember
from stalactite.configs import VFLConfig
from stalactite.helpers import reporting, run_local_agents


from airflow.decorators import task
from airflow import DAG
from airflow.models import Variable
from airflow.models.baseoperator import chain

from stalactite.configs import VFLConfig
from utils.prepare_mnist import load_data as load_mnist
from utils.prepare_sbol_smm import load_data as load_sbol

formatter = logging.Formatter(
    fmt='[%(asctime)s] [%(name)s] [%(levelname)s] > %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
StreamHandler = logging.StreamHandler(stream=sys.stdout)
StreamHandler.setFormatter(formatter)
logging.basicConfig(handlers=[StreamHandler], level=logging.DEBUG)
logging.getLogger("urllib3").setLevel(logging.CRITICAL)

logger = logging.getLogger("airflow-logger")


@task
def make_data_preparation(config: VFLConfig):
    """
    This task is used to prepare data & train-test split for the Airflow experiment
    :return:
    """
    logger.info(f"make data preparation...")
    if config.data.dataset.lower() == "mnist":
        load_mnist(config.data.host_path_data_dir, config.common.world_size, binary=True)
    elif config.data.dataset.lower() == "sbol":
        load_sbol(config.data.host_path_data_dir, config.common.world_size)
    else:
        raise ValueError(f"unknown dataset: {config.data.dataset.lower()}")
    logger.info(f"data preparation SUCCESS")


@task
def get_processor(config: VFLConfig):
    dataset = {}
    for m in range(config.common.world_size):
        dataset[m] = datasets.load_from_disk(
            os.path.join(f"{config.data.host_path_data_dir}/part_{m}")
        )

    image_preprocessor = ImagePreprocessorEff \
        if config.vfl_model.vfl_model_name == "efficientnet" else ImagePreprocessor

    processors = [
        image_preprocessor(dataset=dataset[i], member_id=i, params=config) for i, v in dataset.items()
    ]

    master_processor = image_preprocessor(dataset=datasets.load_from_disk(
        os.path.join(f"{config.data.host_path_data_dir}/master_part")
    ), member_id=-1, params=config, is_master=True)

    with open("/opt/airflow/data/processors_dict.pkl", 'wb') as f:
        pickle.dump({"processors": processors, "master_processor": master_processor}, f)


def run(config: VFLConfig, inference=False):
    with open("/opt/airflow/data/processors_dict.pkl", 'rb') as f:
        loaded_preprocessors = pickle.load(f)
    processors = loaded_preprocessors["processors"]
    master_processor = loaded_preprocessors["master_processor"]
    if inference:
        config.vfl_model.do_train = False
    with reporting(config):
        shared_party_info = dict()
        if 'logreg' in config.vfl_model.vfl_model_name:
            master_class = HonestPartyMasterLogReg
            member_class = HonestPartyMemberLogReg
        elif "resnet" in config.vfl_model.vfl_model_name:
            master_class = HonestPartyMasterResNetSplitNN
            member_class = HonestPartyMemberResNet
        elif "efficientnet" in config.vfl_model.vfl_model_name:
            master_class = HonestPartyMasterEfficientNetSplitNN
            member_class = HonestPartyMemberEfficientNet
        elif "mlp" in config.vfl_model.vfl_model_name:
            master_class = HonestPartyMasterMLPSplitNN
            member_class = HonestPartyMemberMLP
        else:
            member_class = HonestPartyMemberLinReg
            if config.vfl_model.is_consequently:
                master_class = HonestPartyMasterLinRegConsequently
            else:
                master_class = HonestPartyMasterLinReg
        master = master_class(
            uid="master",
            epochs=config.vfl_model.epochs,
            report_train_metrics_iteration=config.common.report_train_metrics_iteration,
            report_test_metrics_iteration=config.common.report_test_metrics_iteration,
            processor=master_processor,
            target_uids=master_processor.dataset[config.data.train_split][config.data.uids_key][
                        :config.data.dataset_size],
            inference_target_uids=master_processor.dataset[config.data.test_split][config.data.uids_key],
            batch_size=config.vfl_model.batch_size,
            eval_batch_size=config.vfl_model.eval_batch_size,
            model_update_dim_size=0,
            run_mlflow=config.master.run_mlflow,
            do_train=config.vfl_model.do_train,
            do_predict=config.vfl_model.do_predict,
            model_name=config.vfl_model.vfl_model_name if
            config.vfl_model.vfl_model_name in ["resnet", "mlp", "efficientnet"] else None,
            model_params=config.master.master_model_params
        )

        member_ids = [f"member-{member_rank}" for member_rank in range(config.common.world_size)]

        members = [
            member_class(
                uid=member_uid,
                member_record_uids=processors[member_rank].dataset[config.data.train_split][config.data.uids_key],
                member_inference_record_uids=processors[member_rank].dataset[config.data.test_split][
                    config.data.uids_key],
                model_name=config.vfl_model.vfl_model_name,
                processor=processors[member_rank],
                batch_size=config.vfl_model.batch_size,
                eval_batch_size=config.vfl_model.eval_batch_size,
                epochs=config.vfl_model.epochs,
                report_train_metrics_iteration=config.common.report_train_metrics_iteration,
                report_test_metrics_iteration=config.common.report_test_metrics_iteration,
                is_consequently=config.vfl_model.is_consequently,
                members=member_ids if config.vfl_model.is_consequently else None,
                do_train=config.vfl_model.do_train,
                do_predict=config.vfl_model.do_predict,
                do_save_model=config.vfl_model.do_save_model,
                model_path=config.vfl_model.vfl_model_path,
                model_params=config.member.member_model_params,
                use_inner_join=True if member_rank == 0 else False

            )
            for member_rank, member_uid in enumerate(member_ids)
        ]

        def local_master_main():
            logger.info("Starting thread %s" % threading.current_thread().name)
            comm = LocalMasterPartyCommunicator(
                participant=master,
                world_size=config.common.world_size,
                shared_party_info=shared_party_info,
                recv_timeout=config.master.recv_timeout,
            )
            comm.run()
            logger.info("Finishing thread %s" % threading.current_thread().name)

        def local_member_main(member: PartyMember):
            logger.info("Starting thread %s" % threading.current_thread().name)
            comm = LocalMemberPartyCommunicator(
                participant=member,
                world_size=config.common.world_size,
                shared_party_info=shared_party_info,
                recv_timeout=config.member.recv_timeout,
            )
            comm.run()
            logger.info("Finishing thread %s" % threading.current_thread().name)

        run_local_agents(
            master=master, members=members, target_master_func=local_master_main, target_member_func=local_member_main
        )


@task
def train(config: VFLConfig):
    logger.info(f"train for {config.vfl_model.vfl_model_name}")
    run(config=config, inference=False)
    logger.info(f"train for model: {config.vfl_model.vfl_model_name} SUCCESS")


@task
def infer(config: VFLConfig):
    logger.info(f"inference for {config.vfl_model.vfl_model_name}")
    run(config=config, inference=True)
    logger.info(f"inference for model: {config.vfl_model.vfl_model_name} SUCCESS")


def build_dag(
        dag_id: str,
        model_names: List[str],
        dataset_name: str
):
    with DAG(
            dag_id=dag_id,
            schedule=timedelta(days=10086),
            start_date=pendulum.datetime(2021, 1, 1, tz="UTC"),
            catchup=False,
    ) as dag:

        if dataset_name == "mnist":
            config = VFLConfig.load_and_validate("/opt/airflow/dags/configs/linreg-mnist-local.yml")
        elif dataset_name == "sbol":
            config = VFLConfig.load_and_validate("/opt/airflow/dags/configs/logreg-sbol-local.yml")
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        data_preparation = make_data_preparation(config=config)
        processor = get_processor(config=config)

        train_infer_list = []
        for model_name in model_names:
            config.vfl_model.vfl_model_name = model_name
            train_infer_list.append(train(config=config))
            train_infer_list.append(infer(config=config))

        chain(*train_infer_list)

        data_preparation >> processor >> train_infer_list[0]

    return dag


mnist_dag = build_dag(dag_id="mnist_dag", model_names=["linreg", "efficientnet"], dataset_name="mnist")
sbol_dag = build_dag(dag_id="sbol_dag", model_names=["logreg", "mlp", "resnet"], dataset_name="mnist")
#todo: homecredit dag
#todo: avito dag
