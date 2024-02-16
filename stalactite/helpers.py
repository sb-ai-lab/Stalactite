from contextlib import contextmanager
from threading import Thread
from typing import List, Callable, Optional

import mlflow

from stalactite.base import PartyMaster, PartyMember
from stalactite.configs import VFLConfig
from stalactite.ml.arbitered.base import PartyArbiter


@contextmanager
def reporting(config: VFLConfig):
    if config.master.run_mlflow:
        mlflow.set_tracking_uri(f"http://{config.prerequisites.mlflow_host}:{config.prerequisites.mlflow_port}")
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
