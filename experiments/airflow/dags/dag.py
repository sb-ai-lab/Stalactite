import os
import random
import sys
import pickle
import uuid
from typing import List, Callable
import sys
import logging

import pendulum
from datetime import timedelta
import pickle
from functools import partial
import functools

from stalactite.party_single_impl import (PartySingleLogregMulticlass, PartySingleLogreg, PartySingleMLP,
                                          PartySingleResNet)
from stalactite.configs import VFLConfig
import threading
from typing import Optional
import datasets
import mlflow
import optuna
from optuna.study import MaxTrialsCallback
from optuna.trial import TrialState

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
from utils.utils import (suggest_params, rsetattr, change_master_model_param, change_member_model_param,
                         compute_hidden_layers, metrics_to_opt_dict)


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
    elif config.data.dataset.lower() in ["sbol", "sbol_smm", "sbol_zvuk", "sbol_smm_zvuk"]:
        use_smm = True if config.data.dataset.lower() == "sbol_smm" else False
        load_sbol(data_dir_path=config.data.host_path_data_dir, parts_num=config.common.world_size, use_smm=use_smm,
                  sample=config.data.dataset_size, seed=config.common.seed)
    elif config.data.dataset.lower() in ["home_credit_bureau_pos", "home_credit", "home_credit_bureau", "home_credit_pos"]:
        use_bureau = True if config.data.dataset.lower() == "home_credit_bureau" else False
        load_home_credit(data_dir_path=config.data.host_path_data_dir, parts_num=config.common.world_size,
                         use_bureau=use_bureau, sample=config.data.dataset_size, seed=config.common.seed)
    else:
        raise ValueError(f"unknown dataset: {config.data.dataset.lower()}")
    logger.info(f"data preparation SUCCESS")


@task
def dump_processor(config: VFLConfig, processors_dict_path: str):
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


def objective_func(trial, config):
    processors_path = "/opt/airflow/dags/dags_data/processors_dict.pkl"
    processors, master_processor = load_processors(processors_path)
    if config.data.dataset_size == -1:
        config.data.dataset_size = len(master_processor.dataset[config.data.train_split][config.data.uids_key])
    suggested_params = suggest_params(trial=trial, config=config)

    for param_name, param_val in suggested_params.items():
        if param_name in ["batch_size", "learning_rate", "weight_decay"]:
            rsetattr(config, f"vfl_model.{param_name}", param_val)
            if param_name == "batch_size":
                # do eval every 0.2 epoch
                report_metrics_iteration = config.data.dataset_size // config.vfl_model.batch_size // 5
                if report_metrics_iteration < 1:
                    report_metrics_iteration = 1
                rsetattr(config, f"common.report_train_metrics_iteration", report_metrics_iteration)
                rsetattr(config, f"common.report_test_metrics_iteration", report_metrics_iteration)
        elif param_name == "dropout":
            change_member_model_param(config=config, model_param_name=param_name, new_value=param_val)

        elif param_name == "first_hidden_coef":
            assert type(suggested_params["layers_num"]) is int
            hidden_layers = compute_hidden_layers(config=config, suggested_params=suggested_params, param_val=param_val)
            change_member_model_param(config=config, model_param_name="hidden_channels", new_value=hidden_layers)
            change_master_model_param(config=config, model_param_name="input_dim", new_value=hidden_layers[-1])

        elif param_name == "hidden_factor":
            assert type(suggested_params["resnet_block_num"]) is int
            hid_factor = [param_val for _ in range(suggested_params["resnet_block_num"])]
            change_member_model_param(config=config, model_param_name="hid_factor", new_value=hid_factor)
        elif param_name in ["layers_num", "resnet_block_num"]:
            pass
        else:
            raise ValueError(f"Unsupported param type: {param_name}")

    # for param_name, param_val in suggested_params.items():
    #         rsetattr(config, f"vfl_model.{param_name}", param_val)
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
            model_params=config.master.master_model_params,
            seed=config.common.seed
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
                use_inner_join=True if member_rank == 0 else False,
                seed=config.common.seed

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

        metrics_to_optimize = metrics_to_opt_dict[config.data.dataset]
        current_run = mlflow.active_run()
        runs = mlflow.search_runs(experiment_names=["airflow"])
        metric = runs[runs["run_id"] == str(current_run.info.run_id)].iloc[0][metrics_to_optimize]
        # metric = random.random() #todo: remove

    return metric


def objective_func_single(trial, config):
    #todo: make similar to obj func
    processors_path = "/opt/airflow/dags/dags_data/processors_dict.pkl"
    processors, master_processor = load_processors(processors_path)
    if config.data.dataset_size == -1:
        config.data.dataset_size = len(processors[0].dataset[config.data.train_split][config.data.uids_key])
    suggested_params = suggest_params(trial=trial, config=config)
    # todo: revise it
    for param_name, param_val in suggested_params.items():
        if param_name in ["batch_size", "learning_rate", "weight_decay"]:
            rsetattr(config, f"vfl_model.{param_name}", param_val)
            if param_name == "batch_size":
                # do eval every 0.2 epoch
                report_metrics_iteration = config.data.dataset_size // config.vfl_model.batch_size // 5
                if report_metrics_iteration < 1:
                    report_metrics_iteration = 1
                rsetattr(config, f"common.report_train_metrics_iteration", report_metrics_iteration)
                rsetattr(config, f"common.report_test_metrics_iteration", report_metrics_iteration)
        elif param_name == "dropout":
            change_member_model_param(config=config, model_param_name=param_name, new_value=param_val)

        elif param_name == "first_hidden_coef":
            assert type(suggested_params["layers_num"]) is int
            hidden_layers = compute_hidden_layers(config=config, suggested_params=suggested_params, param_val=param_val)
            change_member_model_param(config=config, model_param_name="hidden_channels", new_value=hidden_layers)
            change_master_model_param(config=config, model_param_name="input_dim", new_value=hidden_layers[-1])

        elif param_name == "hidden_factor":
            assert type(suggested_params["resnet_block_num"]) is int
            hid_factor = [param_val for _ in range(suggested_params["resnet_block_num"])]
            change_member_model_param(config=config, model_param_name="hid_factor", new_value=hid_factor)
        elif param_name in ["layers_num", "resnet_block_num"]:
            pass
        else:
            raise ValueError(f"Unsupported param type: {param_name}")

    with (reporting(config)):
        if 'logreg' in config.vfl_model.vfl_model_name:
            if config.data.dataset.lower() == "mnist":
                single_party_class = PartySingleLogregMulticlass

            elif config.data.dataset.lower() in ["sbol", "sbol_smm", "home_credit_bureau_pos", "home_credit"]:
                single_party_class = PartySingleLogreg
            else:
                raise ValueError(f"unknown dataset: {config.data.dataset.lower()} for logreg model")

        elif "mlp" in config.vfl_model.vfl_model_name:
            single_party_class = PartySingleMLP
        elif "resnet" in config.vfl_model.vfl_model_name:
            single_party_class = PartySingleResNet
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

        metrics_to_optimize = metrics_to_opt_dict[config.data.dataset]
        current_run = mlflow.active_run()
        runs = mlflow.search_runs(experiment_names=["airflow"])
        metric = runs[runs["run_id"] == str(current_run.info.run_id)].iloc[0][metrics_to_optimize]
        return metric

def run_opt(config, n_trials: int):
    study = optuna.create_study(direction="maximize")
    objective = partial(objective_func, config=config)
    study.optimize(objective, n_trials=n_trials)
    print("Best hyperparameters:", study.best_params)
    print("Best value:", study.best_value)



def run_opt_parallel(config, n_trials: int, study_uid: str, obj_func: Callable):
    study = optuna.create_study(direction="maximize",
                                study_name=study_uid,
                                storage="postgresql+psycopg2://dmitriy:dmitriy@postgres2/dmitriy",
                                load_if_exists=True
                                )
    objective = partial(obj_func, config=config)
    study.optimize(objective, n_trials=n_trials,
                   callbacks=[MaxTrialsCallback(n_trials, states=(TrialState.COMPLETE,))],)
    print("Best hyperparameters:", study.best_params)
    print("Best value:", study.best_value)


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
            single_party_class = PartySingleMLP
        elif "resnet" in config.vfl_model.vfl_model_name:
            single_party_class = PartySingleResNet
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
def train_opt(config: VFLConfig, n_trials: int):
    logger.info(f"train for {config.vfl_model.vfl_model_name}")
    run_opt(config=config, n_trials=n_trials)
    logger.info(f"train-opt for model: {config.vfl_model.vfl_model_name} SUCCESS")


@task
def train_opt_parallel(config: VFLConfig, n_trials: int, obj_func: Callable):
    logger.info(f"train for {config.vfl_model.vfl_model_name}")
    study_uid = get_study_uuid(
        model_name=config.vfl_model.vfl_model_name,
        dataset_name=config.data.dataset,
        world_size=config.common.world_size
    )
    run_opt_parallel(config=config, n_trials=n_trials, study_uid=study_uid, obj_func=obj_func)
    logger.info(f"train-opt for model: {config.vfl_model.vfl_model_name} SUCCESS")


@task
def infer(config: VFLConfig):
    logger.info(f"inference for {config.vfl_model.vfl_model_name}")
    run(config=config)
    logger.info(f"inference for model: {config.vfl_model.vfl_model_name} SUCCESS")


def get_config(dataset_name: str, model_name: str, is_single: bool = False, members: int = None) -> VFLConfig:
    postfix = "single" if is_single else "vfl"
    config = VFLConfig.load_and_validate(
        f"/opt/airflow/dags/configs/{model_name}/{postfix}/{model_name}-{dataset_name}-{postfix}.yml"
    )
    if members is not None:
        config.common.world_size = members
        config.data.host_path_data_dir = config.data.host_path_data_dir + str(members)
    return config


@task
def dump_study_uuid(model_name: str, dataset_name: str, world_size: int):
    pickle_path = f"/opt/airflow/dags/dags_data/{model_name}_{dataset_name}_{world_size}.pkl"
    study_uid = str(uuid.uuid4())
    with open(pickle_path, 'wb') as f:
        pickle.dump({"uid": study_uid}, f)


def get_study_uuid(model_name: str, dataset_name: str, world_size: int):
    pickle_path = f"/opt/airflow/dags/dags_data/{model_name}_{dataset_name}_{world_size}.pkl"
    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)
    return data["uid"]


def build_dag(
        dag_id: str,
        model_names: List[str],
        dataset_name: str,
        world_sizes: List[int],
        n_trials: int = None,
        n_jobs: int = 1,
):
    with DAG(
            dag_id=dag_id,
            schedule=timedelta(days=10086),
            start_date=pendulum.datetime(2021, 1, 1, tz="UTC"),
            catchup=False,
    ) as dag:

        processors_path = "/opt/airflow/dags/dags_data/processors_dict.pkl"
        data_preparators, get_processor_tasks, study_uids_task, train_tasks = [], [], [], []

        for world_size in world_sizes:
            for model_name in model_names:
                config = get_config(dataset_name=dataset_name, model_name=model_name, members=world_size,
                                    is_single=False)
                data_preparators.append(make_data_preparation(config=config))
                get_processor_tasks.append(dump_processor(config=config, processors_dict_path=processors_path))
                # tasks.append(train(config=config))
                study_uids_task.append(dump_study_uuid(
                    model_name=model_name, dataset_name=dataset_name, world_size=world_size)
                )

                # tasks.append(train_opt(config=config, n_trials=n_trials))
                model_ds_train_tasks = []
                for job in range(n_jobs):
                    model_ds_train_tasks.append(
                        train_opt_parallel(config=config, n_trials=n_trials, obj_func=objective_func)
                    )
                train_tasks.append(model_ds_train_tasks)

        seq_num = len(model_names)*len(world_sizes)
        for i in range(seq_num):
            if i == seq_num - 1:
                data_preparators[i] >> get_processor_tasks[i] >> study_uids_task[i] >> train_tasks[i]
            else:
                data_preparators[i] >> get_processor_tasks[i] >> study_uids_task[i] >> train_tasks[i] >> data_preparators[i+1]

    return dag


def build_single_mode_dag(dag_id: str,
                          models_names_list: List,
                          dataset_names_list: List,
                          n_trials: int = None,
                          n_jobs: int = 1,
                          ):
    with DAG(
            dag_id=dag_id,
            schedule=timedelta(days=10086),
            start_date=pendulum.datetime(2021, 1, 1, tz="UTC"),
            catchup=False,
    ) as dag:

        processors_path = "/opt/airflow/dags/dags_data/"

        data_preparators, get_processor_tasks, study_uids_task, train_tasks = [], [], [], []

        for model_name in models_names_list:
            for dataset_name in dataset_names_list:
                "processors_dict.pkl"
                config = get_config(dataset_name=dataset_name, model_name=model_name, is_single=True)
                data_preparators.append(make_data_preparation(config=config))
                get_processor_tasks.append(dump_processor(config=config, processors_dict_path=processors_path))
                # tasks.append(train(config=config))
                study_uids_task.append(dump_study_uuid(
                    model_name=model_name, dataset_name=dataset_name, world_size=1)
                )

                model_ds_train_tasks = []
                for job in range(n_jobs):
                    model_ds_train_tasks.append(
                        train_opt_parallel(config=config, n_trials=n_trials, obj_func=objective_func_single)
                    )
                train_tasks.append(model_ds_train_tasks)
                # data_preparators.append(make_data_preparation(config=config))
                #
                # config = get_config(dataset_name=dataset_name, model_name=model_name, is_single=True)
                # # save processor
                # tasks.append(dump_processor(
                #     config=config,
                #     processors_dict_path=processors_path))
                # # load processor and do train-infer
                # tasks.append(train_infer_single(
                #     config=config,
                #     processors_dict_path=processors_path))

        seq_num = len(models_names_list) * len(dataset_names_list)
        for i in range(seq_num):
            if i == seq_num - 1:
                data_preparators[i] >> get_processor_tasks[i] >> study_uids_task[i] >> train_tasks[i]
            else:
                data_preparators[i] >> get_processor_tasks[i] >> study_uids_task[i] >> train_tasks[i] >> \
                data_preparators[i + 1]

    return dag


# single_dag = build_single_mode_dag(
#     dag_id="single_dag",
#     models_names_list=["logreg", "mlp", "resnet"],
#     dataset_names_list=["mnist", "sbol_smm", "sbol", "home_credit_bureau_pos", "home_credit"]
# )


mnist_dag = build_dag(dag_id="mnist_dag", model_names=["logreg", "mlp"], dataset_name="mnist",
                             world_sizes=[2, 3], n_trials=2, n_jobs=2)


sbol_smm_dag = build_dag(dag_id="sbol_smm_dag", model_names=["logreg", "mlp", "resnet"],
                         dataset_name="sbol_smm", world_sizes=[2], n_trials=30, n_jobs=4)

sbol_zvuk_dag = build_dag(dag_id="sbol_zvuk_dag", model_names=["logreg", "mlp", "resnet"],
                          dataset_name="sbol_zvuk", world_sizes=[2], n_trials=30, n_jobs=4)


sbol_smm_zvuk_dag = build_dag(dag_id="sbol_smm_zvuk_dag", model_names=["logreg", "mlp", "resnet"],
                         dataset_name="sbol_smm_zvuk", world_sizes=[3], n_trials=30, n_jobs=4)


home_credit_dag = build_dag(dag_id="home_credit_dag", model_names=["logreg", "mlp", "resnet"],
                            dataset_name="home_credit_bureau_pos", world_sizes=[3], n_trials=30, n_jobs=4)

single_home_credit_dag = build_single_mode_dag(dag_id="single_home_credit_dag", models_names_list=["logreg", "mlp", "resnet"],
                            dataset_names_list=["home_credit"], n_trials=2, n_jobs=2)

single_sbol_dag = build_single_mode_dag(
    dag_id="single_sbol_dag",
    models_names_list=["logreg", "mlp", "resnet"],
    dataset_names_list=["sbol"], n_trials=30, n_jobs=4
)

# home_credit_dag = build_dag(dag_id="home_credit_dag", model_names=["logreg", "mlp", "resnet"],
#                             dataset_name="home_credit_bureau_pos", world_sizes=[3])
# todo: avito dag