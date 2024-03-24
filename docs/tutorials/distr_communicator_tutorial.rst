.. _distr_comm_tutorial:

*how-to:* Create and launch distributed experiment
====================================================

Launch the prerequisites if you want to log metrics and parameters by following :ref:`prerequisites_tutorial`.

Define the script to run as the command in the agent container.

.. note::
    Before reading this section, please, read the :ref:`local_comm_tutorial` as it explains the definition of the
    PartyMaster and PartyMember instances.


Stalactite CLI launches the containers with the commands defined in ``run_grpc_agent.py``.
Main function creates the master, member (and, if ``grpc_arbiter.use_arbiter`` is set to True, arbiter) participants,
defines the gRPC-based distributed communicators and starts the experiment on each agent.

Firstly, we implement several helper functions, to get the agents. The ``load_parameters`` functions were
defined in the :ref:`local_comm_tutorial`, we import them from the examples folder.

.. warning::
    The ``load_parameters`` executes the custom preprocessing functions for ``mnist`` and ``sbol`` datasets in case when
    processed data do not exist. We highly recommend to pass the path to the processed datasets when you run distributed
    experiments. This section is undergoing active development, and we resolve it in the nearest future.

.. code-block:: python

    from examples.utils.local_experiment import load_processors as load_processors_honest
    from examples.utils.local_arbitered_experiment import load_processors as load_processors_arbitered





    def get_party_master(config_path: str, is_infer: bool = False) -> PartyMaster:
        config = VFLConfig.load_and_validate(config_path)
        if config.grpc_arbiter.use_arbiter:
            master_processor, processors = load_processors_arbitered(config)
            master_processor = master_processor if config.data.dataset.lower() == "sbol_smm" else processors[0]
            master_class = ArbiteredPartyMasterLogReg
            if config.grpc_arbiter.security_protocol_params is not None:
                # If security protocol parameters are passed, we initialize the SecurityProtocolPaillier with them
                if config.grpc_arbiter.security_protocol_params.he_type == 'paillier':
                    sp_agent = SecurityProtocolPaillier(**config.grpc_arbiter.security_protocol_params.init_params)
                else:
                    raise ValueError('Only paillier HE implementation is available')
            else:
                sp_agent = None
            return master_class(
                uid="master",
                epochs=config.vfl_model.epochs,
                report_train_metrics_iteration=config.common.report_train_metrics_iteration,
                report_test_metrics_iteration=config.common.report_test_metrics_iteration,
                processor=master_processor,
                target_uids=master_processor.dataset[config.data.train_split][config.data.uids_key][
                            :config.data.dataset_size],
                inference_target_uids=master_processor.dataset[config.data.test_split][config.data.uids_key],
                batch_size=config.vfl_model.batch_size,
                eval_batch_size=config.vfl_model.eval_batch_size,
                model_update_dim_size=0,
                run_mlflow=config.master.run_mlflow,
                num_classes=config.data.num_classes,
                security_protocol=sp_agent,
                do_train=not is_infer, # To launch from stalactite CLI we use separate commands for training and inference
                do_predict=is_infer, # To launch from stalactite CLI we use separate commands for training and inference
                do_save_model=config.vfl_model.do_save_model,
                model_path=config.vfl_model.vfl_model_path,
            )

        else:
            master_processor, processors = load_processors_honest(config)
            if 'logreg' in config.vfl_model.vfl_model_name:
                master_class = HonestPartyMasterLogReg
            elif "resnet" in config.vfl_model.vfl_model_name:
                master_class = HonestPartyMasterResNetSplitNN
            elif "efficientnet" in config.vfl_model.vfl_model_name:
                master_class = HonestPartyMasterEfficientNetSplitNN
            elif "mlp" in config.vfl_model.vfl_model_name:
                master_class = HonestPartyMasterMLPSplitNN
            else:
                master_class = HonestPartyMasterLinReg
            return master_class(
                uid="master",
                epochs=config.vfl_model.epochs,
                report_train_metrics_iteration=config.common.report_train_metrics_iteration,
                report_test_metrics_iteration=config.common.report_test_metrics_iteration,
                processor=master_processor,
                target_uids=master_processor.dataset[config.data.train_split][config.data.uids_key][
                            :config.data.dataset_size],
                inference_target_uids=master_processor.dataset[config.data.test_split][config.data.uids_key],
                batch_size=config.vfl_model.batch_size,
                eval_batch_size=config.vfl_model.eval_batch_size,
                model_update_dim_size=0,
                run_mlflow=config.master.run_mlflow,
                do_train=not is_infer, # To launch from stalactite CLI we use separate commands for training and inference
                do_predict=is_infer, # To launch from stalactite CLI we use separate commands for training and inference
                model_name=config.vfl_model.vfl_model_name if
                config.vfl_model.vfl_model_name in ["resnet", "mlp", "efficientnet"] else None,
                model_params=config.master.master_model_params
            )

    # Because we create separate containers we pass the rank to load correct processors
    def get_party_member(config_path: str, member_rank: int, is_infer: bool = False) -> PartyMember:
        config = VFLConfig.load_and_validate(config_path)
        if config.grpc_arbiter.use_arbiter:
            master_processor, processors = load_processors_arbitered(config)
            member_class = ArbiteredPartyMemberLogReg
            if config.grpc_arbiter.security_protocol_params is not None:
                if config.grpc_arbiter.security_protocol_params.he_type == 'paillier':
                    sp_agent = SecurityProtocolPaillier(**config.grpc_arbiter.security_protocol_params.init_params)
                else:
                    raise ValueError('Only paillier HE implementation is available')
            else:
                sp_agent = None

            return member_class(
                uid=f"member-{member_rank}",
                member_record_uids=processors[member_rank].dataset[config.data.train_split][config.data.uids_key],
                member_inference_record_uids=processors[member_rank].dataset[config.data.test_split][
                    config.data.uids_key],
                processor=processors[member_rank],
                batch_size=config.vfl_model.batch_size,
                eval_batch_size=config.vfl_model.eval_batch_size,
                epochs=config.vfl_model.epochs,
                report_train_metrics_iteration=config.common.report_train_metrics_iteration,
                report_test_metrics_iteration=config.common.report_test_metrics_iteration,
                num_classes=config.data.num_classes,
                security_protocol=sp_agent,
                do_train=not is_infer, # To launch from stalactite CLI we use separate commands for training and inference
                do_predict=is_infer, # To launch from stalactite CLI we use separate commands for training and inference
                do_save_model=config.vfl_model.do_save_model,
                model_path=config.vfl_model.vfl_model_path,
                use_inner_join=False
            )

        else:
            master_processor, processors = load_processors_honest(config)
            if 'logreg' in config.vfl_model.vfl_model_name:
                member_class = HonestPartyMemberLogReg
            elif "resnet" in config.vfl_model.vfl_model_name:
                member_class = HonestPartyMemberResNet
            elif "efficientnet" in config.vfl_model.vfl_model_name:
                member_class = HonestPartyMemberEfficientNet
            elif "mlp" in config.vfl_model.vfl_model_name:
                member_class = HonestPartyMemberMLP
            else:
                member_class = HonestPartyMemberLinReg

            return member_class(
                uid=f"member-{member_rank}",
                member_record_uids=processors[member_rank].dataset[config.data.train_split][config.data.uids_key],
                member_inference_record_uids=processors[member_rank].dataset[config.data.test_split][config.data.uids_key],
                model_name=config.vfl_model.vfl_model_name,
                processor=processors[member_rank],
                batch_size=config.vfl_model.batch_size,
                eval_batch_size=config.vfl_model.eval_batch_size,
                epochs=config.vfl_model.epochs,
                report_train_metrics_iteration=config.common.report_train_metrics_iteration,
                report_test_metrics_iteration=config.common.report_test_metrics_iteration,
                is_consequently=config.vfl_model.is_consequently,
                members=None,
                do_predict=is_infer,
                do_train=not is_infer,
                do_save_model=config.vfl_model.do_save_model,
                model_path=config.vfl_model.vfl_model_path,
                model_params=config.member.member_model_params,
                use_inner_join=True if member_rank == 0 else False
            )

    # For the arbitered setting, we must initialize the arbiter instance with the corresponding parameters
    def get_party_arbiter(config_path: str, is_infer: bool = False) -> PartyArbiter:
        config = VFLConfig.load_and_validate(config_path)
        if not config.grpc_arbiter.use_arbiter:
            raise RuntimeError('Arbiter should not be called in honest setting.')

        arbiter_class = PartyArbiterLogReg
        if config.grpc_arbiter.security_protocol_params is not None:
            if config.grpc_arbiter.security_protocol_params.he_type == 'paillier':
                # Note, that arbiter holds the SecurityProtocolArbiterPaillier protocol class, which differs from the
                # members` and master`s SP by additional functionality of the trusted party
                sp_arbiter = SecurityProtocolArbiterPaillier(**config.grpc_arbiter.security_protocol_params.init_params)
            else:
                raise ValueError('Only paillier HE implementation is available')
        else:
            sp_arbiter = None

        return arbiter_class(
            uid="arbiter",
            epochs=config.vfl_model.epochs,
            batch_size=config.vfl_model.batch_size,
            eval_batch_size=config.vfl_model.eval_batch_size,
            security_protocol=sp_arbiter,
            learning_rate=config.vfl_model.learning_rate,
            momentum=0.0,
            num_classes=config.data.num_classes,
            do_predict=is_infer,
            do_train=not is_infer,
        )


The CLI ``stalactite local --multi-process start`` and ``stalactite <master/member/arbiter> start`` commands launches
containers using the ``grpc-base:latest`` image built from one of the dockerfiles which can be found in the
``docker/`` folder in `github <https://github.com/sb-ai-lab/vfl-benchmark/tree/main>`_.

To define the ``command`` argument which will be used on the containers start, we need to implement two scripts,
running the master and member communicators.

For the master, member and arbiter communicators the following script is used (``run_grpc_agent.py``):

.. code-block:: python

    import os

    import click

    from stalactite.communications.distributed_grpc_comm import (
        GRpcMasterPartyCommunicator,
        GRpcArbiterPartyCommunicator,
        GRpcMemberPartyCommunicator,
    )
    from stalactite.helpers import reporting
    from stalactite.configs import VFLConfig
    from stalactite.ml.arbitered.base import Role
    from stalactite.data_utils import get_party_master, get_party_arbiter, get_party_member

    import logging

    logger = logging.getLogger(__name__)


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
        # Load the configuration file
        config = VFLConfig.load_and_validate(config_path)
        arbiter_grpc_host = None
        if config.grpc_arbiter.use_arbiter:
            # If we launch containers in the multiprocess regime, we assign the hostname to the arbiter container
            # and pass the container hostname as the environmental variable,
            # Otherwise, in the multihost environment we need to pass the arbiter container host explicitly through
            # the config
            arbiter_grpc_host = os.environ.get("GRPC_ARBITER_HOST", config.grpc_arbiter.container_host)

        if role == Role.member:
            # We pass the rank as the env variable to the container
            member_rank = int(os.environ.get("RANK", 0))
            # Same to the arbiter host logic is applied to the master container host variable.
            grpc_host = os.environ.get("GRPC_SERVER_HOST", config.master.container_host)
            # GRpcMemberPartyCommunicator requires additional keyword args to act as the gRPC client to the
            # server on master
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
        elif role == Role.master:
            # Due to the metrics and parameters are logged from the master, we do not need to start the mlflow
            # experiment for each agent
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
                    arbiter_host=arbiter_grpc_host,
                    arbiter_port=config.grpc_arbiter.port if config.grpc_arbiter.use_arbiter else None,
                    use_arbiter=config.grpc_arbiter.use_arbiter,
                    sent_task_timout=config.member.sent_task_timout,
                )

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
                host=config.grpc_arbiter.host,
                server_thread_pool_size=config.grpc_server.server_threadpool_max_workers,
                max_message_size=config.grpc_arbiter.max_message_size,
                logging_level=config.grpc_arbiter.logging_level,
                rendezvous_timeout=config.common.rendezvous_timeout,
                recv_timeout=config.grpc_arbiter.recv_timeout,
            )
        else:
            raise ValueError(f'Unknown role to initialize communicator ({role}). '
                             f'Role must be one of `master`, `member`, `arbiter`')

        # Start the communicator, which will launch the gRPC server (for master / arbiter) and run the participant
        comm.run()


    if __name__ == "__main__":
        main()



After everything is set, the distributed experiment can be launched, now you can run the distributed experiments using
Stalactite CLI.