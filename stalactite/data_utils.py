# TODO: this file is added temporary. It will be removed or significantly changed after refactoring of the preprocessors
from stalactite.data_preprocessors import ImagePreprocessor, TabularPreprocessor

import os
from typing import Any, Optional
import datasets

from stalactite.configs import VFLConfig
from stalactite.party_member_impl import PartyMemberImpl
from stalactite.party_master_impl import PartyMasterImpl, PartyMasterImplConsequently, PartyMasterImplLogreg


# TODO : add prerocessing of the datasets
def load_processors(config_path: str) -> list[Any]:
    config = VFLConfig.load_and_validate(config_path)

    if config.data.dataset.lower() == "mnist":
        dataset = {}
        for m in range(config.common.world_size):
            dataset[m] = datasets.load_from_disk(
                os.path.join(f"{config.data.host_path_data_dir}/part_{m}")
            )

        processors = [
            ImagePreprocessor(dataset=dataset[i], member_id=i, params=config) for i, v in dataset.items()
        ]

    elif config.data.dataset.lower() == "sbol":

        dataset = {}

        for m in range(config.common.world_size):
            dataset[m] = datasets.load_from_disk(
                os.path.join(f"{config.data.host_path_data_dir}/part_{m}")
            )
        processors = [
            TabularPreprocessor(dataset=dataset[i], member_id=i, params=config) for i, v in dataset.items()
        ]

    else:
        raise ValueError(f"Unknown dataset: {config.data.dataset}, choose one from ['mnist', 'multilabel']")

    return processors


def get_party_master(config_path: str):
    processors = load_processors(config_path)
    config = VFLConfig.load_and_validate(config_path)
    target_uids = [str(i) for i in range(config.data.dataset_size)]
    if 'logreg' in config.common.vfl_model_name:
        master_class = PartyMasterImplLogreg
    else:
        if config.common.is_consequently:
            master_class = PartyMasterImplConsequently
        else:
            master_class = PartyMasterImpl
    return master_class(
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


def get_party_member(config_path: str, member_rank: int):
    config = VFLConfig.load_and_validate(config_path)
    processors = load_processors(config_path)
    target_uids = [str(i) for i in range(config.data.dataset_size)]
    return PartyMemberImpl(
        uid=f"member-{member_rank}",
        member_record_uids=target_uids,
        model_name=config.common.vfl_model_name,
        processor=processors[member_rank],
        batch_size=config.common.batch_size,
        epochs=config.common.epochs,
        report_train_metrics_iteration=config.common.report_train_metrics_iteration,
        report_test_metrics_iteration=config.common.report_test_metrics_iteration,
    )
