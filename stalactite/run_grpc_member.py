import os

import click

from stalactite.communications import GRpcMemberPartyCommunicator
from stalactite.configs import VFLConfig
from stalactite.data_utils import get_party_member


@click.command()
@click.option('--config-path', type=str, default='../configs/config.yml')
def main(config_path):
    member_rank = int(os.environ.get('RANK', 0))
    config = VFLConfig.load_and_validate(config_path)

    grpc_host = os.environ.get('GRPC_SERVER_HOST', config.master.container_host)

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
