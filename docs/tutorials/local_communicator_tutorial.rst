.. _local_comm_tutorial:

*how-to:* Create and launch local experiment
============================================

First of all, you need to access your configuration parameters. For this purpose, the ``VFLConfig`` Pydantic model is
used. Import it and load the parameters by running

.. code-block:: python

    import logging
    from pathlib import Path
    import threading

    from stalactite.configs import VFLConfig

    def run(config_path: str):
        config = VFLConfig.load_and_validate(config_path)

Then, define the structure of the MlFlow run:

.. code-block:: python

    import mlflow

    def run(config_path: str):
        ...
        if config.master.run_mlflow:
            mlflow.set_tracking_uri(f"http://{config.prerequisites.mlflow_host}:{config.prerequisites.mlflow_port}")
            mlflow.set_experiment(config.common.experiment_label)
            mlflow.start_run()

        # Experiment goes here
        ...

        if config.master.run_mlflow:
                mlflow.end_run()


For the experiment you will need the preprocessors to pass into the agents. For this purpose, we
define the ``load_processors`` function

.. code-block:: python

    import os

    from stalactite.data_preprocessors import ImagePreprocessor, TabularPreprocessor
    from examples.utils.prepare_mnist import load_data as load_mnist
    from examples.utils.prepare_sbol_smm import load_data as load_sbol_smm

    def load_processors(config_path: str):
        config = VFLConfig.load_and_validate(config_path)

        if config.data.dataset.lower() == "mnist":

            if not os.path.exists(config.data.host_path_data_dir):
                load_mnist(config.data.host_path_data_dir, config.common.world_size)

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
            if not os.path.exists(config.data.host_path_data_dir):
                load_sbol_smm(os.path.dirname(config.data.host_path_data_dir), parts_num=2)

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

    def run(config_path: str):
        ...
        model_name = config.common.vfl_model_name
        processors = load_processors(config_path)
        # Processors prepare and contain data for each agent

        # We initialize the target uids here because we want to simulate only partially available data
        target_uids = [str(i) for i in range(config.data.dataset_size)]
        # Local communicator requires party information, we initialize it as an empty dictionary as no data is passed for
        # the experiment
        shared_party_info = dict()
        ...


After we can get all required data, let's initialize the master class

.. code-block:: python

    from stalactite.party_master_impl import PartyMasterImpl, PartyMasterImplConsequently, PartyMasterImplLogreg

    def run(config_path: str):
        ...
        if 'logreg' in config.common.vfl_model_name:
            master_class = PartyMasterImplLogreg
        else:
            if config.common.is_consequently:
                master_class = PartyMasterImplConsequently
            else:
                master_class = PartyMasterImpl
        master = master_class(
            uid="master",
            epochs=config.common.epochs,
            report_train_metrics_iteration=config.common.report_train_metrics_iteration,
            report_test_metrics_iteration=config.common.report_test_metrics_iteration,
            processor=processors[0], # For the master we take the first processor
            target_uids=target_uids,
            batch_size=config.common.batch_size,
            model_update_dim_size=0, # Let us leave this parameter as is, it will be updated later
            run_mlflow=config.master.run_mlflow,
        )
        ....

After the master is ready, we need to prepare the members:

.. code-block:: python

    from stalactite.party_member_impl import PartyMemberImpl
    def run(config_path: str):
        ...
        # Members ids are required before the initialization only in local sequential linear regression case
        # for the batcher initialization (it needs to have a list of the participants),
        # and are not applicable or used in other cases

        member_ids = [f"member-{member_rank}" for member_rank in range(config.common.world_size)]

        members = [
            PartyMemberImpl(
                uid=member_uid,
                member_record_uids=target_uids,
                model_name=config.common.vfl_model_name,
                processor=processors[member_rank],
                batch_size=config.common.batch_size,
                epochs=config.common.epochs,
                report_train_metrics_iteration=config.common.report_train_metrics_iteration,
                report_test_metrics_iteration=config.common.report_test_metrics_iteration,
                is_consequently=config.common.is_consequently,
                members=member_ids if config.common.is_consequently else None,
            )
            for member_rank, member_uid in enumerate(member_ids)
        ]
        ...

The local experiment is launched in one Python process in different threads, thus we need to create thread target
functions to run member and master. Within those functions we will initialize and run the local communicator class to
facilitate operations between master and members.

.. code-block:: python

    import logging
    from stalactite.communications.local import LocalMasterPartyCommunicator, LocalMemberPartyCommunicator

    logger = logging.getLogger(__name__)

    def run(config_path: str):
        ...
        def local_master_main():
            logger.info("Starting thread %s" % threading.current_thread().name)
            comm = LocalMasterPartyCommunicator(
                participant=master,
                world_size=config.common.world_size,
                shared_party_info=shared_party_info
            )
            comm.run()
            logger.info("Finishing thread %s" % threading.current_thread().name)

        def local_member_main(member: PartyMember):
            logger.info("Starting thread %s" % threading.current_thread().name)
            # We need to pass the `master_id` into local communicator only. In distributed case,
            # members identify the master in the rendezvous.
            comm = LocalMemberPartyCommunicator(
                participant=member,
                world_size=config.common.world_size,
                shared_party_info=shared_party_info,
                master_id=master.id
            )
            comm.run()
            logger.info("Finishing thread %s" % threading.current_thread().name)
        ...

Now we can finalize the `run` by starting and joining the threads.

.. code-block:: python

    from threading import Thread

    def run(config_path: str):
        ...

        threads = [
            Thread(name=f"main_{master.id}", daemon=True, target=local_master_main),
            *(
                Thread(
                    name=f"main_{member.id}",
                    daemon=True,
                    target=local_member_main,
                    args=(member,)
                )
                for member in members
            )
        ]

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

The full example is available in our `github <https://github.com/sb-ai-lab/vfl-benchmark/tree/main>`_ at
``examples/utils/local_experiment.py``.
