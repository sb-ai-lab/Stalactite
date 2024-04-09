import os

import click

from stalactite.communications.distributed_grpc_comm import (
    GRpcMasterPartyCommunicator,
    GRpcArbiterPartyCommunicator,
    GRpcMemberPartyCommunicator,
)
from stalactite.helpers import reporting, global_logging
from stalactite.configs import VFLConfig
from stalactite.ml.arbitered.base import Role
from stalactite.data_utils import get_party_master, get_party_arbiter, get_party_member

import logging

logging.getLogger('fsspec').setLevel(logging.ERROR)
logging.getLogger('git').setLevel(logging.ERROR)


@click.command()
@click.option("--config-path", type=str, default="../configs/config.yml")
@click.option(
    "--infer",
    is_flag=True,
    show_default=True,
    default=False,
    help="Run in an inference mode.",
)
@click.option(
    "--role",
    type=str,
    required=True,
    help="Role of the agent in the experiment (one of `master`, `arbiter`, `member`).",
)
def main(config_path, infer, role):
    config = VFLConfig.load_and_validate(config_path)
    global_logging(role=role, config=config)

    arbiter_grpc_host = None
    if config.grpc_arbiter.use_arbiter:
        arbiter_grpc_host = os.environ.get("GRPC_ARBITER_HOST", config.grpc_arbiter.external_host)

    if role == Role.member:
        member_rank = int(os.environ.get("RANK", 0))
        grpc_host = os.environ.get("GRPC_SERVER_HOST", config.master.external_host)
        comm = GRpcMemberPartyCommunicator(
            participant=get_party_member(config_path, member_rank, is_infer=infer),
            master_host=grpc_host,
            master_port=config.grpc_server.port,
            max_message_size=config.grpc_server.max_message_size,
            logging_level=config.member.logging_level,
            heartbeat_interval=config.member.heartbeat_interval,
            sent_task_timout=config.member.sent_task_timout,
            rendezvous_timeout=config.common.rendezvous_timeout,
            recv_timeout=config.member.recv_timeout,
            arbiter_host=arbiter_grpc_host,
            arbiter_port=config.grpc_arbiter.port if config.grpc_arbiter.use_arbiter else None,
            use_arbiter=config.grpc_arbiter.use_arbiter,
        )
        comm.run()

    elif role == Role.master:
        with reporting(config):
            comm = GRpcMasterPartyCommunicator(
                participant=get_party_master(config_path, is_infer=infer),
                world_size=config.common.world_size,
                port=config.grpc_server.port,
                server_thread_pool_size=config.grpc_server.server_threadpool_max_workers,
                max_message_size=config.grpc_server.max_message_size,
                logging_level=config.master.logging_level,
                prometheus_server_port=config.prerequisites.prometheus_server_port,
                run_prometheus=config.master.run_prometheus,
                experiment_label=config.common.experiment_label,
                rendezvous_timeout=config.common.rendezvous_timeout,
                disconnect_idle_client_time=config.master.disconnect_idle_client_time,
                time_between_idle_connections_checks=config.master.time_between_idle_connections_checks,
                recv_timeout=config.master.recv_timeout,
                arbiter_host=arbiter_grpc_host,
                arbiter_port=config.grpc_arbiter.port if config.grpc_arbiter.use_arbiter else None,
                use_arbiter=config.grpc_arbiter.use_arbiter,
                sent_task_timout=config.member.sent_task_timout,
            )
            comm.run()

    elif role == Role.arbiter:
        if not config.grpc_arbiter.use_arbiter:
            raise ValueError(
                'Configuration parameter `grpc_arbiter.use_arbiter` is set to False, you should not '
                'initialize Arbiter in this experiment'
            )
        comm = GRpcArbiterPartyCommunicator(
            participant=get_party_arbiter(config_path, is_infer=infer),
            world_size=config.common.world_size,
            port=config.grpc_arbiter.port,
            server_thread_pool_size=config.grpc_server.server_threadpool_max_workers,
            max_message_size=config.grpc_arbiter.max_message_size,
            logging_level=config.grpc_arbiter.logging_level,
            rendezvous_timeout=config.common.rendezvous_timeout,
            recv_timeout=config.grpc_arbiter.recv_timeout,
        )
        comm.run()

    else:
        raise ValueError(f'Unknown role to initialize communicator ({role}). '
                         f'Role must be one of `master`, `member`, `arbiter`')


if __name__ == "__main__":
    main()
