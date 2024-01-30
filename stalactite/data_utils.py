# TODO: this file is added temporary. It will be removed or significantly changed after refactoring of the preprocessors


from stalactite.models.logreg_batch import LogisticRegressionBatch
from stalactite.data_loader import load, init, DataPreprocessor
from stalactite.data_preprocessors import ImagePreprocessor, TabularPreprocessor

import os
import pickle
import random
import uuid
from dataclasses import dataclass
from typing import Any, Optional, Union
import datasets

import numpy as np
import torch

from stalactite.base import DataTensor
from stalactite.configs import VFLConfig
from stalactite.data_loader import AttrDict
from stalactite.models.linreg_batch import LinearRegressionBatch
from stalactite.party_master_impl import PartyMasterImpl
from stalactite.party_member_impl import PartyMemberImpl
from stalactite.party_master_impl import PartyMasterImpl, PartyMasterImplConsequently, PartyMasterImplLogreg


# @dataclass
# class MasterData:
#     # targets: DataTensor
#     processors: Any
#     target_uids: list[str]
#     # test_targets: DataTensor
#
#
# @dataclass
# class MemberData:
#     model_update_dim_size: int
#     params: Any
#     processors: Any
#     member_uids: list[str]
#     # dataset: Any

@dataclass
class PartyData:
    config: VFLConfig
    processors: Any
    target_uids: list[str]
    input_dims_list: list[list[int]]
    members_datasets_uids: list[list[str]]
    classes_idx: Optional[list[int]] = None


def load_and_prepare_datasets(config_path: str) -> PartyData:
    config = VFLConfig.load_and_validate(config_path)

    model_name = config.common.vfl_model_name

    if 'logreg' in model_name:
        # todo: hide it somehow
        classes_idx = [x for x in range(19)]
    else:
        classes_idx = list()
    n_labels = len(classes_idx)

    if config.data.dataset.lower() == "mnist":
        params = init(config_path=os.path.abspath(config_path))
        dataset = {}

        for m in range(config.common.world_size):
            dataset[m] = datasets.load_from_disk(
                os.path.join(f"{config.data.host_path_data_dir}/part_{m}")
            )
        processors = [
            ImagePreprocessor(dataset=dataset[i], member_id=i, data_params=params[i].data) for i, v in dataset.items()
        ]
        input_dims_list = [[619], [392, 392], [204, 250, 165], [], [108, 146, 150, 147, 68]]

    elif config.data.dataset.lower() == "sbol":
        smm = "smm_" if config.data.use_smm else ""
        dim = 1356 if smm == "smm_" else 1345
        # input_dims_list = [[0], [1345, 11]]

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

    num_dataset_records = [200 + random.randint(100, 1000) for _ in range(config.common.world_size)]
    shared_record_uids = [str(i) for i in range(config.data.dataset_size)]
    target_uids = shared_record_uids
    members_datasets_uids = [
        [*shared_record_uids, *(str(uuid.uuid4()) for _ in range(num_records - len(shared_record_uids)))]
        for num_records in num_dataset_records
    ]

    return PartyData(
        processors=processors,
        config=config,
        target_uids=target_uids,
        input_dims_list=input_dims_list,
        members_datasets_uids=members_datasets_uids,
        classes_idx=classes_idx,
    )


def get_party_master(config_path: str):
    party_data = load_and_prepare_datasets(config_path)
    config = party_data.config

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
        processor=party_data.processors[0],
        target_uids=party_data.target_uids,
        batch_size=config.common.batch_size,
        model_update_dim_size=0,
        run_mlflow=config.master.run_mlflow,
    )


def get_party_member(config_path: str, member_rank: int):
    party_data = load_and_prepare_datasets(config_path)
    config = party_data.config

    if 'logreg' in config.common.vfl_model_name:
        model = LogisticRegressionBatch(
            input_dim=party_data.input_dims_list[config.common.world_size - 1][member_rank],
            output_dim=len(party_data.classes_idx),
            learning_rate=config.common.learning_rate,
            class_weights=None,
            init_weights=0.005
        )
    else:
        model = LinearRegressionBatch(
            input_dim=party_data.input_dims_list[config.common.world_size - 1][member_rank],
            output_dim=1,
            reg_lambda=0.2
        )

    return PartyMemberImpl(
        uid=f"member-{member_rank}",
        model_update_dim_size=party_data.input_dims_list[config.common.world_size - 1][member_rank],
        member_record_uids=party_data.members_datasets_uids[member_rank],
        model=model,
        processor=party_data.processors[member_rank],
        batch_size=config.common.batch_size,
        epochs=config.common.epochs,
        report_train_metrics_iteration=config.common.report_train_metrics_iteration,
        report_test_metrics_iteration=config.common.report_test_metrics_iteration,
    )

#
#
# def load_parameters(config_path: str):
#
#     config = VFLConfig.load_and_validate(config_path)
#     sample_size = int(os.environ.get("SAMPLE_SIZE", 10000))
#
#     # models input dims for 1, 2, 3 and 5 members
#     if config.data.dataset.lower() == "mnist":
#         input_dims_list = [[619], [392, 392], [204, 250, 165], [], [108, 146, 150, 147, 68]]
#         # params = init(config_path=os.path.join(BASE_PATH, "experiments/configs/config_local_mnist.yaml"))
#         params = init(config_path=os.path.abspath(config_path))
#
#     for m in range(config.common.world_size):
#         if config.common.vfl_model_name == "linreg":
#             params[m].data.dataset = f"mnist_binary38_parts{config.common.world_size}"
#         elif config.common.vfl_model_name == "logreg" or "catboost":
#             params[m].data.dataset = f"mnist_binary01_38_parts{config.common.world_size}"
#         else:
#             raise ValueError("Unknown model name {}".format(config.common.vfl_model_name))
#
#     if config.data.dataset.lower() == "mnist":
#         dataset, _ = load(params)
#         # todo: add processor here
#         processors = [ImagePreprocessor(dataset=dataset[i], member_id=i, data_params=params[i].data) for i, v in dataset.items()]
#
#     else:
#         raise ValueError(f"Unknown dataset: {config.data.dataset}, choose one from ['mnist', 'multilabel']")
#
#     return input_dims_list, params, processors
#
# def prepare_data(config_path: str, config: VFLConfig, master: bool, member_rank: Optional[int] = None) -> Union[MasterData, MemberData]:
#     model_name = config.common.vfl_model_name
#
#
#
#     random.seed(config.data.random_seed)
#     np.random.seed(config.data.random_seed)
#     torch.manual_seed(config.data.random_seed)
#     torch.cuda.manual_seed_all(config.data.random_seed)
#     torch.backends.cudnn.deterministic = True
#
#     # with open(
#     #     os.path.join(
#     #         os.path.expanduser(config.data.host_path_data_dir), f"datasets_{config.common.world_size}_members.pkl"
#     #     ),
#     #     "rb",
#     # ) as f:
#     #     datasets_list = pickle.load(f)["data"]
#     input_dims_list, params, processors = load_parameters(config_path)
#
#     shared_record_uids = [str(i) for i in range(config.data.dataset_size)]
#     if master:
#         target_uids = shared_record_uids
#         # targets = datasets_list[0]["train_train"]["label"][: config.data.dataset_size]
#         # test_targets = datasets_list[0]["train_val"]["label"]
#         return MasterData(processors=processors, target_uids=target_uids)
#
#     else:
#         input_dims_list = [[619], [304, 315], [204, 250, 165], [], [108, 146, 150, 147, 68]]
#         num_dataset_records = [200 + random.randint(100, 1000) for _ in range(config.common.world_size)]
#         members_datasets_uids = [
#             [*shared_record_uids, *(str(uuid.uuid4()) for _ in range(num_records - len(shared_record_uids)))]
#             for num_records in num_dataset_records
#         ]
#         return MemberData(
#             model_update_dim_size=input_dims_list[config.common.world_size - 1][member_rank],
#             params=params,
#             processors=processors,
#             member_uids=members_datasets_uids[member_rank],
#             # dataset=datasets_list[member_rank],
#         )
#
#
# def get_party_master(
# config_path: str,
#     config: VFLConfig,
# ):
#     master_data = prepare_data(config_path, config, master=True)
#
#     # return PartyMasterImpl(
#     #     uid="master",
#     #     epochs=config.common.epochs,
#     #     report_train_metrics_iteration=config.common.report_train_metrics_iteration,
#     #     report_test_metrics_iteration=config.common.report_test_metrics_iteration,
#     #     target=master_data.targets,
#     #     test_target=master_data.test_targets,
#     #     target_uids=master_data.target_uids,
#     #     batch_size=config.common.batch_size,
#     #     model_update_dim_size=0,
#     #     run_mlflow=config.master.run_mlflow,
#     # )
#
#     if 'logreg' in config.common.vfl_model_name:
#         master_class = PartyMasterImplLogreg
#     else:
#         if config.common.is_consequently:
#             master_class = PartyMasterImplConsequently
#         else:
#             master_class = PartyMasterImpl
#     return master_class(
#         uid="master",
#         epochs=config.common.epochs,
#         report_train_metrics_iteration=config.common.report_train_metrics_iteration,
#         report_test_metrics_iteration=config.common.report_test_metrics_iteration,
#         processor=master_data.processors[0],
#         # target=targets,
#         # test_target=test_targets,
#         target_uids=master_data.target_uids,
#         batch_size=config.common.batch_size,
#         model_update_dim_size=0,
#         run_mlflow=config.master.run_mlflow,
#     )
#
#
# def get_party_member(config_path: str, config: VFLConfig, member_rank: int):
#
#     member_data = prepare_data(config_path, config, master=False, member_rank=member_rank)
#     # params = {
#     #     member_rank: {
#     #         "common": {"random_seed": config.data.random_seed, "parties_num": config.common.world_size},
#     #         "data": {
#     #             "dataset_part_prefix": "part_",
#     #             "train_split": "train_train",
#     #             "test_split": "train_val",
#     #             "features_key": f"image_part_{member_rank}",
#     #             "label_key": "label",
#     #         },
#     #     },
#     #     "joint_config": True,
#     #     "parties": list(range(config.common.world_size)),
#     # }
#     # params = AttrDict(params)
#     # params[member_rank].data.dataset = f"mnist_binary38_parts{config.common.world_size}"
#
#     # return PartyMemberImpl(
#     #     uid=f"member-{member_rank}",
#     #     model_update_dim_size=member_data.model_update_dim_size,
#     #     member_record_uids=member_data.member_uids,
#     #     model=LinearRegressionBatch(input_dim=member_data.model_update_dim_size, output_dim=1, reg_lambda=0.2),
#     #     dataset=member_data.dataset,
#     #     data_params=params[member_rank].data,
#     #     batch_size=config.common.batch_size,
#     #     epochs=config.common.epochs,
#     #     report_train_metrics_iteration=config.common.report_train_metrics_iteration,
#     #     report_test_metrics_iteration=config.common.report_test_metrics_iteration,
#     # )
#     model_name = config.common.vfl_model_name
#
#     if 'logreg' in model_name:
#         classes_idx = [x for x in range(19)]
#         # remove classes with low positive targets rate
#         classes_idx = [x for x in classes_idx if x not in [18, 3, 12, 14]]
#     else:
#         classes_idx = list()
#     n_labels = len(classes_idx)
#
#
#     if 'logreg' in config.common.vfl_model_name:
#         model = lambda member_rank: LogisticRegressionBatch(
#             input_dim=member_data.model_update_dim_size,
#             output_dim=n_labels,
#             learning_rate=config.common.learning_rate,
#             class_weights=None,
#             init_weights=0.005
#         )
#     else:
#         model = lambda member_rank: LinearRegressionBatch(
#             input_dim=member_data.model_update_dim_size,
#             output_dim=1,
#             reg_lambda=0.2
#         )
#
#     return PartyMemberImpl(
#             uid=f"member-{member_rank}",
#             model_update_dim_size=member_data.model_update_dim_size,
#             member_record_uids=member_data.member_uids,
#             model=model(member_rank),
#             # dataset=datasets_list[member_rank],
#             # data_params=params[member_rank]['data'],
#             processor=member_data.processors[member_rank],
#             batch_size=config.common.batch_size,
#             epochs=config.common.epochs,
#             report_train_metrics_iteration=config.common.report_train_metrics_iteration,
#             report_test_metrics_iteration=config.common.report_test_metrics_iteration,
#         )
