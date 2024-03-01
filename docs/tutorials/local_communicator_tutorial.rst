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

Then, define the structure of the MlFlow run by calling the context manager:

.. code-block:: python

    from stalactite.helpers import reporting

    def run(config_path: str):
        ...
        with reporting(config):
            # Experiment goes here
            ...

For the experiment you will need the preprocessors to pass into the agents. For this purpose, we
define the ``load_processors`` function

.. code-block:: python

    import os

    from stalactite.data_preprocessors import ImagePreprocessor, TabularPreprocessor
    from examples.utils.prepare_mnist import load_data as load_mnist
    from examples.utils.prepare_sbol_smm import load_data as load_sbol_smm

    def load_processors(config):
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
        processors = load_processors(config)
        # Processors prepare and contain data for each agent

        # We initialize the target uids here because we want to simulate only partially available data
        target_uids = [str(i) for i in range(config.data.dataset_size)]
        inference_target_uids = [str(i) for i in range(500)]
        # Local communicator requires party information, we initialize it as an empty dictionary as no data is passed for
        # the experiment
        shared_party_info = dict()
        ...


After we can get all required data, let's initialize the master class

.. code-block:: python

    from stalactite.ml import (
        HonestPartyMasterLinRegConsequently,
        HonestPartyMasterLinReg,
        HonestPartyMemberLogReg,
        HonestPartyMemberLinReg,
        HonestPartyMasterLogReg
    )

    def run(config_path: str):
        ...
        if 'logreg' in config.vfl_model.vfl_model_name:
            master_class = HonestPartyMasterLogReg
            member_class = HonestPartyMemberLogReg
        else:
            member_class = HonestPartyMemberLinReg
            if config.vfl_model.is_consequently:
                master_class = HonestPartyMasterLinRegConsequently
            else:
                master_class = HonestPartyMasterLinReg

        master = master_class(
            uid="master",
            epochs=config.vfl_model.epochs,
            report_train_metrics_iteration=config.common.report_train_metrics_iteration,
            report_test_metrics_iteration=config.common.report_test_metrics_iteration,
            processor=processors[0], # For the master we take the first processor
            target_uids=target_uids,
            inference_target_uids=inference_target_uids,
            batch_size=config.vfl_model.batch_size,
            eval_batch_size=config.vfl_model.eval_batch_size,
            model_update_dim_size=0, # Let us leave this parameter as is, it will be updated later
            run_mlflow=config.master.run_mlflow,
            do_train=config.vfl_model.do_train,
            do_predict=config.vfl_model.do_predict,
        )
        ....

After the master is ready, we need to prepare the members:

.. code-block:: python

    from stalactite.party_member_impl import PartyMemberImpl
    def run(config_path: str):
        ...
        # Members ids are required before the initialization only in local sequential linear regression case
        # for the make_batcher initialization (it needs to have a list of the participants),
        # and are not applicable or used in other cases

        member_ids = [f"member-{member_rank}" for member_rank in range(config.common.world_size)]

        members = [
            member_class(
                uid=member_uid,
                member_record_uids=target_uids,
                member_inference_record_uids=inference_target_uids,
                model_name=config.vfl_model.vfl_model_name,
                processor=processors[member_rank],
                batch_size=config.vfl_model.batch_size,
                eval_batch_size=config.vfl_model.eval_batch_size,
                epochs=config.vfl_model.epochs,
                report_train_metrics_iteration=config.common.report_train_metrics_iteration,
                report_test_metrics_iteration=config.common.report_test_metrics_iteration,
                is_consequently=config.vfl_model.is_consequently,
                members=member_ids if config.vfl_model.is_consequently else None,
                do_train=config.vfl_model.do_train,
                do_predict=config.vfl_model.do_predict,
                do_save_model=config.vfl_model.do_save_model,
                model_path=config.vfl_model.vfl_model_path
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

Now we can finalize the `run` by starting and joining the threads using the utility function ``run_local_agents``.

.. code-block:: python

    from stalactite.helpers import run_local_agents

    def run(config_path: str):
        ...

        run_local_agents(
            master=master,
            members=members,
            target_master_func=local_master_main,
            target_member_func=local_member_main
        )

The full example is available in our `github <https://github.com/sb-ai-lab/vfl-benchmark/tree/main>`_ at
``examples/utils/local_experiment.py``.
