.. _data_proc_tutorial:

*how-to:* Preprocess data
======================================

First of all, we need to clarify that all datasets need to be pre-processed.

It means that we need to have the master_part of dataset, which consist of ``[uid, labels]`` columns and members parts.
Each member part should contain ``[uid, features_part_X]`` columns.


Examples of such pre-processing are ``examples/utils/prepare_mnist.py`` and ``examples/utils/prepare_sbol_smm.py``.
We use `datasets.DatasetDict <https://huggingface.co/docs/datasets/en/package_reference/main_classes#datasets.DatasetDict>`_
as the main dataset structure.

To launch such a preprocessing you need to import ``load_data`` function and run it:

.. code-block:: python

    from examples.utils.prepare_mnist import load_data as load_mnist
    from examples.utils.prepare_sbol_smm import load_data as load_sbol_smm

    load_mnist(config.data.host_path_data_dir, config.common.world_size, binary=binary)
    load_sbol_smm(os.path.dirname(config.data.host_path_data_dir), parts_num=2)

This ``load_data`` function prepares datasets and saves it to config.data.host_path_data_dir
We should mention than if you want to choose sbol_smm dataset, you need to have sbol and smm datasets in ``config.data.host_path_data_dir``

Data processors
-----------------------

Each PartyMaster and PartyMember object has required argument named ``processor``.
Processors are wrappers over datasets, which have ``fit_transform`` methods inside for processing the data before
training models.
By the moment, we have three types of Processors: ``ImageProcessor``, ``ImageProcEffProcessor`` and ``TabularProcessor``.
You can write a custom Processor for your purposes.

- ImageProcessor transform images to flat tensors and makes normalization.
- ImageProcEffProcessor does the same  but without flatting tensors cause we want to us it as input for CNN-like models.
- TabularProcessor is used for tabular data.

Loading prepared datasets and initializing processor which we need

.. code-block:: python

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


Define the master with the master_processor inside

.. code-block:: python

    master = master_class(
                uid="master",
                epochs=config.vfl_model.epochs,
                report_train_metrics_iteration=config.common.report_train_metrics_iteration,
                report_test_metrics_iteration=config.common.report_test_metrics_iteration,
                processor=master_processor,
               ...
            )

Define the members with the member processors inside

.. code-block:: python

    members = [
                member_class(
                    uid=member_uid,
                    member_record_uids=processors[member_rank].dataset[config.data.train_split][config.data.uids_key],
                    member_inference_record_uids=processors[member_rank].dataset[config.data.test_split][config.data.uids_key],
                    model_name=config.vfl_model.vfl_model_name,
                    processor=processors[member_rank],
                    ...
                )
                for member_rank, member_uid in enumerate(member_ids)
            ]

