.. _distr_comm_tutorial:

how-to: Create and launch distributed experiment
====================================================

Launch the prerequisites if you want to log metrics and parameters by following :ref:`prerequisites_tutorial`.

Define the script to run as the command in the agent container.

.. note::
    Before reading this section, please, read the :ref:`local_comm_tutorial` as it explains the definition of the
    PartyMaster and PartyMember instances.


Stalactite CLI launches the containers with the commands defined in ``run_grpc_master.py`` and ``run_grpc_member.py``.
Main function in those creates the master and member participants, defines the gRPC-based distributed communicators and
starts the experiment on each agent.

Firstly, we implement several helper functions, to get the members and master. The ``load_parameters`` function was
defined in the :ref:`local_comm_tutorial`.

.. code-block:: python

    def get_party_master(config_path: str):
        processors = load_processors(config_path) # Load processors
        config = VFLConfig.load_and_validate(config_path) # Load configuration file
        # Define target uids (simulating only partially available data)
        target_uids = [str(i) for i in range(config.data.dataset_size)]
        # The rest of party master definition is similar to the local example
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

    # Because we create separate containers we pass the rank to load correct processors
    def get_party_member(config_path: str, member_rank: int):
        config = VFLConfig.load_and_validate(config_path)
        processors = load_processors(config_path)
        target_uids = [str(i) for i in range(config.data.dataset_size)]
        # We do not pass members ids due to sequential distributed algorithm cannot be used
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

The CLI ``stalactite local --multi-process start`` and ``stalactite <master/member> start`` commands launches containers
using the ``grpc-base:latest`` image built from one of the dockerfiles which can be found in the ``docker/`` folder in
`github <https://github.com/sb-ai-lab/vfl-benchmark/tree/main>`_.

To define the ``command`` argument which will be used on the containers start, we need to implement two scripts,
running the master and member communicators.

For the master communicator the following script is used (``run_grpc_master.py``):

.. code-block:: python

    import click
    import mlflow

    from stalactite.communications import GRpcMasterPartyCommunicator
    from stalactite.configs import VFLConfig
    from stalactite.data_utils import get_party_master

    # We pass the config_path as the CLI argument into the main function

    @click.command()
    @click.option("--config-path", type=str, default="../configs/config.yml")
    def main(config_path):
        # Same to the local experiment load the configuration into the VFLConfig Pydantic model
        config = VFLConfig.load_and_validate(config_path)

        # Define the mlflow run for metrics logging (if enabled)
        if config.master.run_mlflow:
            mlflow.set_tracking_uri(f"http://{config.prerequisites.mlflow_host}:{config.prerequisites.mlflow_port}")
            mlflow.set_experiment(config.common.experiment_label)
            mlflow.start_run()

        # In the GRpcMasterPartyCommunicator several keyword arguments appear, mostly required for the gRPC server start
        comm = GRpcMasterPartyCommunicator(
            participant=get_party_master(config_path),
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
        # Start the communicator, which will launch the gRPC server and run the participant
        comm.run()

        # Finish the mlflow run for metrics logging (if enabled)
        if config.master.run_mlflow:
            mlflow.end_run()

    if __name__ == "__main__":
        main()

For the member communicator we implemented the following (``run_grpc_member.py``):

.. code-block:: python

    import os

    import click

    from stalactite.communications import GRpcMemberPartyCommunicator
    from stalactite.configs import VFLConfig
    from stalactite.data_utils import get_party_member


    @click.command()
    @click.option("--config-path", type=str, default="../configs/config.yml")
    def main(config_path):
        # Due to the metrics and parameters are logged from the master, we do not need to start the mlflow
        # experiment here

        # We pass the rank as the env variable to the container
        member_rank = int(os.environ.get("RANK", 0))
        # Load the configuration file
        config = VFLConfig.load_and_validate(config_path)
        # If we launch containers in the multiprocess regime, we assign the hostname to the master container
        # and pass the master container hostname as the environmental variable,
        # Otherwise, in the multihost environment we need to pass the master container host explicitly through
        # the config
        grpc_host = os.environ.get("GRPC_SERVER_HOST", config.master.container_host)

        # Again, GRpcMemberPartyCommunicator requires additional keyword args to act as the gRPC client to the
        # server on master
        comm = GRpcMemberPartyCommunicator(
            participant=get_party_member(config_path, member_rank),
            master_host=grpc_host,
            master_port=config.grpc_server.port,
            max_message_size=config.grpc_server.max_message_size,
            logging_level=config.member.logging_level,
            heartbeat_interval=config.member.heartbeat_interval,
            task_requesting_pings_interval=config.member.task_requesting_pings_interval,
            sent_task_timout=config.member.sent_task_timout,
            rendezvous_timeout=config.common.rendezvous_timeout,
            recv_timeout=config.member.recv_timeout,
        )
        # Start the communicator, which will launch the gRPC server and run the participant
        comm.run()


    if __name__ == "__main__":
        main()

After everything is set, the distributed experiment can be launched, now you can run the distributed experiments using
Stalactite CLI.