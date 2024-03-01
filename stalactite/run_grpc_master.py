import click

from stalactite.communications import GRpcMasterPartyCommunicator
from stalactite.configs import VFLConfig
from stalactite.data_utils import get_party_master
from stalactite.helpers import reporting


@click.command()
@click.option("--config-path", type=str, default="../configs/config.yml")
@click.option(
    "--infer",
    is_flag=True,
    show_default=True,
    default=False,
    help="Run in an inference mode.",
)
def main(config_path, infer):
    config = VFLConfig.load_and_validate(config_path)
    with reporting(config):
        comm = GRpcMasterPartyCommunicator(
            participant=get_party_master(config_path, is_infer=infer),
            world_size=config.common.world_size,
            port=config.grpc_server.port,
            host=config.grpc_server.host,
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
        )
        comm.run()


if __name__ == "__main__":
    main()
