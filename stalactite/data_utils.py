# TODO: this file is added temporary. It will be removed or significantly changed after refactoring of the preprocessors


import os
import pickle
import random
import uuid
from dataclasses import dataclass
from typing import Any, Optional, Union

import numpy as np
import torch

from stalactite.base import DataTensor
from stalactite.configs import VFLConfig
from stalactite.data_loader import AttrDict
from stalactite.models.linreg_batch import LinearRegressionBatch
from stalactite.party_master_impl import PartyMasterImpl
from stalactite.party_member_impl import PartyMemberImpl


@dataclass
class MasterData:
    targets: DataTensor
    target_uids: list[str]
    test_targets: DataTensor


@dataclass
class MemberData:
    model_update_dim_size: int
    member_uids: list[str]
    dataset: Any


def prepare_data(config: VFLConfig, master: bool, member_rank: Optional[int] = None) -> Union[MasterData, MemberData]:
    random.seed(config.data.random_seed)
    np.random.seed(config.data.random_seed)
    torch.manual_seed(config.data.random_seed)
    torch.cuda.manual_seed_all(config.data.random_seed)
    torch.backends.cudnn.deterministic = True

    with open(
        os.path.join(
            os.path.expanduser(config.data.host_path_data_dir), f"datasets_{config.common.world_size}_members.pkl"
        ),
        "rb",
    ) as f:
        datasets_list = pickle.load(f)["data"]

    shared_record_uids = [str(i) for i in range(config.data.dataset_size)]
    if master:
        target_uids = shared_record_uids
        targets = datasets_list[0]["train_train"]["label"][: config.data.dataset_size]
        test_targets = datasets_list[0]["train_val"]["label"]
        return MasterData(targets=targets, target_uids=target_uids, test_targets=test_targets)

    else:
        input_dims_list = [[619], [304, 315], [204, 250, 165], [], [108, 146, 150, 147, 68]]
        num_dataset_records = [200 + random.randint(100, 1000) for _ in range(config.common.world_size)]
        members_datasets_uids = [
            [*shared_record_uids, *(str(uuid.uuid4()) for _ in range(num_records - len(shared_record_uids)))]
            for num_records in num_dataset_records
        ]
        return MemberData(
            model_update_dim_size=input_dims_list[config.common.world_size - 1][member_rank],
            member_uids=members_datasets_uids[member_rank],
            dataset=datasets_list[member_rank],
        )


def get_party_master(
    config: VFLConfig,
):
    master_data = prepare_data(config, master=True)

    return PartyMasterImpl(
        uid="master",
        epochs=config.common.epochs,
        report_train_metrics_iteration=config.common.report_train_metrics_iteration,
        report_test_metrics_iteration=config.common.report_test_metrics_iteration,
        target=master_data.targets,
        test_target=master_data.test_targets,
        target_uids=master_data.target_uids,
        batch_size=config.common.batch_size,
        model_update_dim_size=0,
        run_mlflow=config.master.run_mlflow,
    )


def get_party_member(config: VFLConfig, member_rank: int):
    member_data = prepare_data(config, master=False, member_rank=member_rank)
    params = {
        member_rank: {
            "common": {"random_seed": config.data.random_seed, "parties_num": config.common.world_size},
            "data": {
                "dataset_part_prefix": "part_",
                "train_split": "train_train",
                "test_split": "train_val",
                "features_key": f"image_part_{member_rank}",
                "label_key": "label",
            },
        },
        "joint_config": True,
        "parties": list(range(config.common.world_size)),
    }
    params = AttrDict(params)
    params[member_rank].data.dataset = f"mnist_binary38_parts{config.common.world_size}"

    return PartyMemberImpl(
        uid=f"member-{member_rank}",
        model_update_dim_size=member_data.model_update_dim_size,
        member_record_uids=member_data.member_uids,
        model=LinearRegressionBatch(input_dim=member_data.model_update_dim_size, output_dim=1, reg_lambda=0.2),
        dataset=member_data.dataset,
        data_params=params[member_rank].data,
        batch_size=config.common.batch_size,
        epochs=config.common.epochs,
        report_train_metrics_iteration=config.common.report_train_metrics_iteration,
        report_test_metrics_iteration=config.common.report_test_metrics_iteration,

    )
