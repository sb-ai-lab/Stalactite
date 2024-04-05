.. _local_comm_tutorial:

*how-to:* Create and launch local experiment
=================================================

Honest-but-curious setting
----------------------------------------

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

    def load_processors(config: VFLConfig):
        """

        Assigns parameters to preprocessor class, which is selected depending on the type of dataset: MNIST or SBOL.
        If there is no data to run the experiment, downloads data after preprocessing.

        """
        if config.data.dataset.lower() == "mnist":

            binary = False if config.vfl_model.vfl_model_name == "efficientnet" else True

            if len(os.listdir(config.data.host_path_data_dir)) == 0:
                load_mnist(config.data.host_path_data_dir, config.common.world_size, binary=binary)

            dataset = {}
            for m in range(config.common.world_size):
                dataset[m] = datasets.load_from_disk(
                    os.path.join(f"{config.data.host_path_data_dir}/part_{m}")
                )

            image_preprocessor = ImagePreprocessorEff \
                if config.vfl_model.vfl_model_name == "efficientnet" else ImagePreprocessor

            processors = [
                image_preprocessor(dataset=dataset[i], member_id=i, params=config) for i, v in dataset.items()
            ]

            master_processor = image_preprocessor(dataset=datasets.load_from_disk(
                os.path.join(f"{config.data.host_path_data_dir}/master_part")
            ), member_id=-1, params=config, is_master=True)

        elif config.data.dataset.lower() == "sbol_smm":

            dataset = {}
            if len(os.listdir(config.data.host_path_data_dir)) == 0:
                load_sbol_smm(os.path.dirname(config.data.host_path_data_dir), parts_num=2)

            for m in range(config.common.world_size):
                dataset[m] = datasets.load_from_disk(
                    os.path.join(f"{config.data.host_path_data_dir}/part_{m}")
                )
            processors = [
                TabularPreprocessor(dataset=dataset[i], member_id=i, params=config) for i, v in dataset.items()
            ]
            master_processor = TabularPreprocessor(dataset=datasets.load_from_disk(
                    os.path.join(f"{config.data.host_path_data_dir}/master_part")
                ), member_id=-1, params=config, is_master=True)

        else:
            raise ValueError(f"Unknown dataset: {config.data.dataset}, choose one from ['mnist', 'multilabel']")

        return master_processor, processors


    def run(config_path: str):
        ...
        model_name = config.common.vfl_model_name
        master_processor, processors = load_processors(config)
        # Processors prepare and contain data for each agent

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
        elif "resnet" in config.vfl_model.vfl_model_name:
            master_class = HonestPartyMasterResNetSplitNN
            member_class = HonestPartyMemberResNet
        elif "efficientnet" in config.vfl_model.vfl_model_name:
            master_class = HonestPartyMasterEfficientNetSplitNN
            member_class = HonestPartyMemberEfficientNet
        elif "mlp" in config.vfl_model.vfl_model_name:
            master_class = HonestPartyMasterMLPSplitNN
            member_class = HonestPartyMemberMLP
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
            processor=master_processor,
            target_uids=master_processor.dataset[config.data.train_split][config.data.uids_key][:config.data.dataset_size],
            inference_target_uids=master_processor.dataset[config.data.test_split][config.data.uids_key],
            batch_size=config.vfl_model.batch_size,
            eval_batch_size=config.vfl_model.eval_batch_size,
            model_update_dim_size=0,
            run_mlflow=config.master.run_mlflow,
            do_train=config.vfl_model.do_train,
            do_predict=config.vfl_model.do_predict,
            model_name=config.vfl_model.vfl_model_name if
            config.vfl_model.vfl_model_name in ["resnet", "mlp", "efficientnet"] else None,
            model_params=config.master.master_model_params
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
                members=member_ids if config.vfl_model.is_consequently else None,
                do_train=config.vfl_model.do_train,
                do_predict=config.vfl_model.do_predict,
                do_save_model=config.vfl_model.do_save_model,
                model_path=config.vfl_model.vfl_model_path,
                model_params=config.member.member_model_params,
                use_inner_join=True if member_rank == 0 else False

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


Arbitered setting
----------------------------------------

For the arbitered setting, the script is available at ``examples/utils/local_arbitered_experiment.py``. There are slight
changed, required for the data loaders and initialization of the agents.

.. code-block:: python

    def load_processors(config: VFLConfig):

        master_processor = None
        if config.data.dataset.lower() == "mnist":

            if len(os.listdir(config.data.host_path_data_dir)) == 0:
                load_mnist(config.data.host_path_data_dir, config.common.world_size)

            dataset = {}
            for m in range(config.common.world_size + 1):
                dataset[m] = datasets.load_from_disk(
                    os.path.join(f"{config.data.host_path_data_dir}/part_{m}")
                )

            image_preprocessor = ImagePreprocessorEff \
                if config.vfl_model.vfl_model_name == "efficientnet" else ImagePreprocessor

            processors = [
                image_preprocessor(dataset=dataset[i], member_id=i, params=config) for i, v in dataset.items()
            ]

            master_processor = image_preprocessor(dataset=datasets.load_from_disk(
                os.path.join(f"{config.data.host_path_data_dir}/master_part")
            ), member_id=-1, params=config, is_master=True)


        elif config.data.dataset.lower() == "sbol_smm":

            dataset = {}
            if len(os.listdir(config.data.host_path_data_dir)) == 0:
                load_sbol_smm(os.path.dirname(config.data.host_path_data_dir), parts_num=2)

            # Here we are giving the master 0th part of the data, hence passing 1st part to the member
            for m in range(1, config.common.world_size + 1):
                dataset[m] = datasets.load_from_disk(
                    os.path.join(f"{config.data.host_path_data_dir}/part_{m}")
                )
            processors = [
                TabularPreprocessor(dataset=dataset[i], member_id=i, params=config) for i, v in dataset.items()
            ]
            master_processor = TabularPreprocessor(master_has_features=True, dataset=datasets.load_from_disk(
                os.path.join(f"{config.data.host_path_data_dir}/master_part_arbiter"),
            ), member_id=0, params=config, is_master=True)

        else:
            raise ValueError(f"Unknown dataset: {config.data.dataset}, choose one from ['mnist', 'multilabel']")

        return master_processor, processors

After the initialization and loading of the data and configuration file (same to honest setting), we need to define the
agents classes

.. code-block:: python


    with reporting(config):
        ...
        master_class = ArbiteredPartyMasterLogReg
        member_class = ArbiteredPartyMemberLogReg
        arbiter_class = PartyArbiterLogReg

        # If the configuration parameter `security_protocol_params` is defined, we can initialize the security protocols
        # on agents
        if config.grpc_arbiter.security_protocol_params is not None:
            if config.grpc_arbiter.security_protocol_params.he_type == 'paillier':
                # Additionaly, arbitered protocol holds the private key and is able to decrypt the data
                sp_arbiter = SecurityProtocolArbiterPaillier(**config.grpc_arbiter.security_protocol_params.init_params)
                sp_agent = SecurityProtocolPaillier(**config.grpc_arbiter.security_protocol_params.init_params)
            else:
                raise ValueError('Only paillier HE implementation is available')
        else:
            sp_arbiter, sp_agent = None, None


Initialize the agents:

.. code-block:: python

    arbiter = arbiter_class(
            uid="arbiter",
            epochs=config.vfl_model.epochs,
            batch_size=config.vfl_model.batch_size,
            eval_batch_size=config.vfl_model.eval_batch_size,
            security_protocol=sp_arbiter,
            learning_rate=config.vfl_model.learning_rate,
            momentum=0.0,
            num_classes=config.data.num_classes,
            do_predict=config.vfl_model.do_predict,
            do_train=config.vfl_model.do_train,
        )

    master_processor = master_processor if config.data.dataset.lower() == "sbol_smm" else processors[0]

    master = master_class(
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
        do_predict=config.vfl_model.do_predict,
        do_train=config.vfl_model.do_train,
        do_save_model=config.vfl_model.do_save_model,
        model_path=config.vfl_model.vfl_model_path,
    )

    member_ids = [f"member-{member_rank}" for member_rank in range(config.common.world_size)]

    members = [
        member_class(
            uid=member_uid,
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
            do_predict=config.vfl_model.do_predict,
            do_train=config.vfl_model.do_train,
            do_save_model=config.vfl_model.do_save_model,
            model_path=config.vfl_model.vfl_model_path,
            use_inner_join=False # We always use fillna, as the master hold the 0th data part
        )
        for member_rank, member_uid in enumerate(member_ids)
    ]

The last part, where we launch threads is changed a little bit, the arbiter is added to the launched agents:

.. code-block:: python

        def local_arbiter_main():
            logger.info("Starting thread %s" % threading.current_thread().name)
            comm = ArbiteredLocalPartyCommunicator(
                participant=arbiter,
                world_size=config.common.world_size,
                shared_party_info=shared_party_info,
                recv_timeout=config.grpc_arbiter.recv_timeout,
            )
            comm.run()
            logger.info("Finishing thread %s" % threading.current_thread().name)

        run_local_agents(
            master=master,
            members=members,
            target_master_func=local_master_main,
            target_member_func=local_member_main,
            arbiter=arbiter, # We add arbiter here using the same function as in previous example
            target_arbiter_func=local_arbiter_main,  # We add launching target function here
        )

After initialization of the agents, you can run any example from the example folders by following either
`honest-but-curious`, or `arbitered` strategy (with or without Homomorphic encryption)
