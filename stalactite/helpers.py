import importlib
import inspect
import logging
import os
import time
from contextlib import contextmanager
from threading import Thread
from typing import List, Callable, Optional

import mlflow

from stalactite.base import PartyMaster, PartyMember
from stalactite.configs import VFLConfig
from stalactite.ml.arbitered.base import PartyArbiter
from stalactite.utils import Role


def get_plugin_agent(module_path: str, role: Role):
    agent_path = f"{module_path}.party_{role}"
    target_class = None
    try:
        module = importlib.import_module(agent_path)
    except ModuleNotFoundError as exc:
        raise ValueError(f"No {role} is defined in plugin/. Check correctness of the config file") from exc
    classes = inspect.getmembers(module, inspect.isclass)
    for cls, path in classes:
        if agent_path in path.__module__ and f"party{role}" in cls.lower():
            target_class = cls
    if target_class is not None:
        return getattr(module, target_class)
    else:
        raise NameError(
            f"Defined classes` names violate the naming convention "
            f"(<arbitrary_prefix>Party{role.capitalize()}<arbitrary_postfix>)"
        )


def global_logging(
        role: Optional[Role] = None,
        config: Optional[VFLConfig] = None,
        logging_level: Optional[int] = logging.DEBUG
):
    loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
    for logger in loggers:
        if logger.name.startswith('stalactite'):
            if logger.hasHandlers():
                logger.handlers.clear()
            handler = logging.StreamHandler()
            formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.propagate = False
            if role is not None:
                if role == Role.master:
                    logger.setLevel(config.master.logging_level)
                elif role == Role.member:
                    logger.setLevel(config.member.logging_level)
                elif role == Role.arbiter:
                    logger.setLevel(config.grpc_arbiter.logging_level)
                else:
                    logger.setLevel(logging.INFO)
            else:
                logger.setLevel(logging_level)
        else:
            logger.propagate = False


@contextmanager
def log_timing(name: str, log_func: Callable = print):
    time_start = time.time()
    log_func(f'Started {name}')
    try:
        yield
    finally:
        log_func(f'{name} time: {round(time.time() - time_start, 4)} sec.')


@contextmanager
def reporting(config: VFLConfig):
    if config.master.run_mlflow:
        mlflow_host = os.environ.get(
            'STALACTITE_MLFLOW_URI', f"http://{config.prerequisites.mlflow_host}:{config.prerequisites.mlflow_port}"
        )
        mlflow.set_tracking_uri(mlflow_host)
        mlflow.set_experiment(config.common.experiment_label)
        mlflow.start_run()

        log_params = {
            "ds_size": config.data.dataset_size,
            "batch_size": config.vfl_model.batch_size,
            "epochs": config.vfl_model.epochs,
            "mode": "vfl" if config.common.world_size >= 1 else "local",
            "members_count": config.common.world_size,
            "exp_uid": config.common.experiment_label,
            "is_consequently": config.vfl_model.is_consequently,
            "model_name": config.vfl_model.vfl_model_name,
            "learning_rate": config.vfl_model.learning_rate,
            "weight_decay": config.vfl_model.weight_decay,
            "dataset": config.data.dataset,

        }
        mlflow.log_params(log_params)
    try:
        yield
    finally:
        if config.master.run_mlflow:
            mlflow.end_run()


def run_local_agents(
        master: PartyMaster,
        members: List[PartyMember],
        target_master_func: Callable,
        target_member_func: Callable,
        arbiter: Optional[PartyArbiter] = None,
        target_arbiter_func: Optional[Callable] = None,
):
    threads = [
        Thread(name=f"main_{master.id}", daemon=True, target=target_master_func),
        *(
            Thread(
                name=f"main_{member.id}",
                daemon=True,
                target=target_member_func,
                args=(member,)
            )
            for member in members
        )
    ]

    if arbiter is not None and target_arbiter_func is not None:
        threads.append(Thread(name=f"main_{arbiter.id}", daemon=True, target=target_arbiter_func))

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()
