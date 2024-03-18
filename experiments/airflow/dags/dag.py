import os
import sys
import pickle

from typing import List
import sys
import logging

import pendulum
from datetime import timedelta
import pickle

from stalactite.party_single_impl import (PartySingleLogregMulticlass, PartySingleLogreg, PartySingleMLP,
                                          PartySingleResNet)
from stalactite.configs import VFLConfig
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
from utils.prepare_home_credit import load_data as load_home_credit

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
    is_single = config.common.world_size == 1

    if not os.path.exists(config.data.host_path_data_dir):
        logger.info(f"making directory for data preparation: {config.data.host_path_data_dir}")
        os.mkdir(config.data.host_path_data_dir)

    if len(os.listdir(config.data.host_path_data_dir)) > 0:
        logger.info(f"data preparation: nothing to do. Exiting...")
        return
    if config.data.dataset.lower() == "mnist":
        load_mnist(config.data.host_path_data_dir, config.common.world_size, binary=False, is_single=is_single)
    elif config.data.dataset.lower() in ["sbol", "sbol_smm"]:
        sbol_only = config.data.dataset.lower() == "sbol"
        load_sbol(config.data.host_path_data_dir, config.common.world_size, is_single=is_single, sbol_only=sbol_only)
    elif config.data.dataset.lower() in ["home_credit_bureau_pos", "home_credit"]:
        applications_only = config.data.dataset.lower() == "home_credit"
        load_home_credit(config.data.host_path_data_dir, config.common.world_size, is_single=is_single,
                         application_only=applications_only)
    else:
        raise ValueError(f"unknown dataset: {config.data.dataset.lower()}")
    logger.info(f"data preparation SUCCESS")


@task
def get_processor(config: VFLConfig, processors_dict_path: str):
    dataset = {}
    for m in range(config.common.world_size):
        dataset[m] = datasets.load_from_disk(
            os.path.join(f"{config.data.host_path_data_dir}/part_{m}")
        )
    if config.vfl_model.vfl_model_name == "efficientnet":
        processor_class = ImagePreprocessorEff
    elif config.data.dataset.lower() == "mnist":
        processor_class = ImagePreprocessor
    else:
        processor_class = TabularPreprocessor

    is_single = config.common.world_size == 1
    processors = [
        processor_class(
            dataset=dataset[i], member_id=i, params=config, is_master=is_single, master_has_features=is_single) for i, v in dataset.items()
    ]

    master_processor = None

    if not is_single:
        master_processor = processor_class(dataset=datasets.load_from_disk(
            os.path.join(f"{config.data.host_path_data_dir}/master_part")
        ), member_id=-1, params=config, is_master=True)

    with open(processors_dict_path, 'wb') as f:
        pickle.dump({"processors": processors, "master_processor": master_processor}, f)


def run(config: VFLConfig):
    processors_path = "/opt/airflow/dags/dags_data/processors_dict.pkl"
    processors, master_processor = load_processors(processors_path)
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
            logger.info("Startinsbol_smmg thread %s" % threading.current_thread().name)
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
def train_infer_single(config: VFLConfig, processors_dict_path: str):
    logger.info(f"train and infer SINGLE for {config.vfl_model.vfl_model_name}")
    run_single(config=config, processors_dict_path=processors_dict_path)
    logger.info(f"train and infer SINGLE for: {config.vfl_model.vfl_model_name} SUCCESS")


def load_processors(processors_dict_path: str):
    with open(processors_dict_path, 'rb') as f:
        loaded_preprocessors = pickle.load(f)
    processors = loaded_preprocessors["processors"]
    master_processor = loaded_preprocessors["master_processor"]
    return processors, master_processor


def run_single(config: VFLConfig, processors_dict_path: str):
    processors, master_processor = load_processors(processors_dict_path)

    with (reporting(config)):
        if 'logreg' in config.vfl_model.vfl_model_name:
            if config.data.dataset.lower() == "mnist":
                single_party_class = PartySingleLogregMulticlass

            elif config.data.dataset.lower() in ["sbol", "sbol_smm", "home_credit_bureau_pos", "home_credit"]:
                single_party_class = PartySingleLogreg
            else:
                raise ValueError(f"unknown dataset: {config.data.dataset.lower()} for logreg model")

        elif "mlp" in config.vfl_model.vfl_model_name:
            single_party_class = PartySingleMLP #todo:

        else:
            raise ValueError("Unknown vfl model %s" % config.vfl_model.vfl_model_name)

        party = single_party_class(
            processor=processors[0],
            batch_size=config.vfl_model.batch_size,
            epochs=config.vfl_model.epochs,
            report_train_metrics_iteration=config.common.report_train_metrics_iteration,
            report_test_metrics_iteration=config.common.report_test_metrics_iteration,
            use_mlflow=config.master.run_mlflow,
            target_uids=processors[0].dataset[config.data.train_split][config.data.uids_key][
                        :config.data.dataset_size],
            model_params=config.member.member_model_params
        )

        party.run()


@task
def train(config: VFLConfig):
    logger.info(f"train for {config.vfl_model.vfl_model_name}")
    run(config=config)
    logger.info(f"train for model: {config.vfl_model.vfl_model_name} SUCCESS")


@task
def infer(config: VFLConfig):
    logger.info(f"inference for {config.vfl_model.vfl_model_name}")
    run(config=config)
    logger.info(f"inference for model: {config.vfl_model.vfl_model_name} SUCCESS")


def get_single_config(dataset_name: str) -> VFLConfig:
    if dataset_name == "mnist":
        config = VFLConfig.load_and_validate("/opt/airflow/dags/configs/logreg-mnist-single.yml")
    elif dataset_name == "sbol":
        config = VFLConfig.load_and_validate("/opt/airflow/dags/configs/logreg-sbol-single.yml")
    elif dataset_name == "sbol_smm":
        config = VFLConfig.load_and_validate("/opt/airflow/dags/configs/logreg-sbol_smm-single.yml")
    elif dataset_name == "home_credit":
        config = VFLConfig.load_and_validate("/opt/airflow/dags/configs/logreg-home_credit-single.yml")
    elif dataset_name == "home_credit_bureau_pos":
        config = VFLConfig.load_and_validate("/opt/airflow/dags/configs/logreg-home_credit_bureau_pos-single.yml")
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    return config


def get_vfl_config(dataset_name: str, model_name: str, members: int) -> VFLConfig:

    config = VFLConfig.load_and_validate(f"/opt/airflow/dags/configs/{model_name}-{dataset_name}-vfl.yml")
    config.common.world_size = members
    config.data.host_path_data_dir = config.data.host_path_data_dir + str(members)
    return config


def build_dag(
        dag_id: str,
        model_names: List[str],
        dataset_name: str,
        world_sizes: List[int]
):
    with DAG(
            dag_id=dag_id,
            schedule=timedelta(days=10086),
            start_date=pendulum.datetime(2021, 1, 1, tz="UTC"),
            catchup=False,
    ) as dag:

        processors_path = "/opt/airflow/dags/dags_data/processors_dict.pkl"

        data_preparators, tasks = [], []

        for model_name in model_names:
            for world_size in world_sizes:
                config = get_vfl_config(dataset_name=dataset_name, model_name=model_name, members=world_size)
                data_preparators.append(make_data_preparation(config=config))
                tasks.append(get_processor(config=config, processors_dict_path=processors_path))
                tasks.append(train(config=config))

        chain(*data_preparators)
        chain(*tasks)

        data_preparators[-1] >> tasks[0]

    return dag


def build_single_mode_dag(dag_id: str,
                          models_names_list: List,
                          dataset_names_list: List):
    with DAG(
            dag_id=dag_id,
            schedule=timedelta(days=10086),
            start_date=pendulum.datetime(2021, 1, 1, tz="UTC"),
            catchup=False,
    ) as dag:

        processors_path = "/opt/airflow/dags/dags_data/processors_dict.pkl"
        data_preparators = []

        # make data preparation for each dataset, saving preprocessed dataset
        for dataset_name in set(dataset_names_list):
            data_preparators.append(make_data_preparation(config=get_single_config(dataset_name)))

        # add processor and train-infer for each model
        tasks = []
        for model_name, dataset_name in zip(models_names_list, dataset_names_list):
            config = get_single_config(dataset_name)
            # save processor
            tasks.append(get_processor(
                config=config,
                processors_dict_path=processors_path))
            # load processor and do train-infer
            tasks.append(train_infer_single(
                config=config,
                processors_dict_path=processors_path))

        chain(*data_preparators)
        chain(*tasks)

        data_preparators[-1] >> tasks[0]

    return dag


single_dag = build_single_mode_dag(
    dag_id="single_dag",
    models_names_list=["logreg", "logreg", "logreg", "logreg", "logreg"],
    dataset_names_list=["mnist", "sbol_smm", "sbol", "home_credit_bureau_pos", "home_credit"]
)


sbol_smm_dag = build_dag(dag_id="sbol_smm_dag", model_names=["logreg", "mlp", "resnet"], dataset_name="sbol_smm",
                         world_sizes=[2])

mnist_dag = build_dag(dag_id="mnist_dag", model_names=["logreg", "mlp", "resnet"], dataset_name="mnist",
                      world_sizes=[2, 3, 4])

home_credit_dag = build_dag(dag_id="home_credit_dag", model_names=["logreg", "mlp", "resnet"],
                            dataset_name="home_credit_bureau_pos", world_sizes=[3])
# todo: avito dag
