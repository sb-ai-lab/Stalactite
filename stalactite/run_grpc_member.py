from dataclasses import dataclass
import uuid
import os
import random
import pickle
from typing import Any

import click
import numpy as np
import torch

from stalactite.configs import VFLConfig
from stalactite.communications import GRpcMemberPartyCommunicator
from stalactite.party_member_impl import PartyMemberImpl
from stalactite.models.linreg_batch import LinearRegressionBatch
from stalactite.data_loader import AttrDict


@dataclass
class MemberData:
    model_update_dim_size: int
    member_uids: list[str]
    dataset: Any


def prepare_data(config: VFLConfig, member_rank: int) -> MemberData:
    input_dims_list = [[619], [304, 315], [204, 250, 165], [], [108, 146, 150, 147, 68]]

    with open(
            os.path.join(
                os.path.expanduser(config.data.host_path_data_dir),
                f"datasets_{config.common.world_size}_members.pkl"
            ),
            'rb'
    ) as f:
        datasets_list = pickle.load(f)['data']

    random.seed(config.data.random_seed)
    np.random.seed(config.data.random_seed)
    torch.manual_seed(config.data.random_seed)
    torch.cuda.manual_seed_all(config.data.random_seed)
    torch.backends.cudnn.deterministic = True
    num_dataset_records = [200 + random.randint(100, 1000) for _ in range(config.common.world_size)]
    shared_record_uids = [str(i) for i in range(config.data.dataset_size)]
    members_datasets_uids = [
        [*shared_record_uids, *(str(uuid.uuid4()) for _ in range(num_records - len(shared_record_uids)))]
        for num_records in num_dataset_records
    ]
    return MemberData(
        model_update_dim_size=input_dims_list[config.common.world_size - 1][member_rank],
        member_uids=members_datasets_uids[member_rank],
        dataset=datasets_list[member_rank]
    )


def get_party_member(config: VFLConfig, member_rank: int):
    member_data = prepare_data(config, member_rank)
    params = {
        member_rank: {
            'common': {'random_seed': config.data.random_seed, 'parties_num': config.common.world_size},
            'data': {
                'dataset_part_prefix': 'part_',
                'train_split': 'train_train',
                'test_split': 'train_val',
                'features_key': f'image_part_{member_rank}',
                'label_key': 'label'
            }
        },
        'joint_config': True,
        'parties': list(range(config.common.world_size))
    }
    params = AttrDict(params)
    params[member_rank].data.dataset = f"mnist_binary38_parts{config.common.world_size}"

    return PartyMemberImpl(
        uid=f"member-{member_rank}",
        model_update_dim_size=member_data.model_update_dim_size,
        member_record_uids=member_data.member_uids,
        model=LinearRegressionBatch(input_dim=member_data.model_update_dim_size, output_dim=1, reg_lambda=0.2),
        dataset=member_data.dataset,
        data_params=params[member_rank].data
    )


@click.command()
@click.option('--config-path', type=str, default='../configs/config.yml')
def main(config_path):
    member_rank = int(os.environ.get('RANK', 0))
    config = VFLConfig.load_and_validate(config_path)

    grpc_host = os.environ.get('GRPC_SERVER_HOST', config.grpc_server.host)

    comm = GRpcMemberPartyCommunicator(
        participant=get_party_member(config, member_rank),
        master_host=grpc_host,
        master_port=config.grpc_server.port,
        logging_level=config.member.logging_level,
        heartbeat_interval=config.member.heartbeat_interval,
    )
    comm.run()


if __name__ == '__main__':
    main()
