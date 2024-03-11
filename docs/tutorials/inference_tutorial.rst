.. _inference_tutorial:

*how-to:* Do VFL inference
====================================================

After you trained your models and saved them using the configuration parameters `do_save_model` and `vfl_model_path`
(:ref:`config_tutorial`) you are able to perform an inference in a VFL fashion.

As you have already seen, launch of the processes using Stalactite can be performed either via the usage of the
Stalactite CLI, or directly from the Python scripts which contain the definition of the experiment.
Current tutorial will demonstrate the inference process for both approaches.

Launching inference from the script
------------------------------------------------

When we defined the agents classes in :ref:`local_comm_tutorial`, for each agent holding the VFL model part, we passed
the parameters `eval_batch_size`, `do_predict`, `vfl_model_path`, `inference_target_uids`
(for the master) and `member_inference_record_uids` (for the member).
Those are all the relevant parameters for the inference.
Of course, the `vfl_model_path` must contain saved after the training process models (which must be available to the
agents which trained them). To save the models after training, the `do_save_model` must be set to ``True`` and
`vfl_model_path` should not change.

When you start the Python process(es) from the script(s), you can do training and inference while running the same
experiment (with the `do_train` and `do_predict` set to ``True``). It will firstly train the model, then save it to the
model path, load it for the inference and report the metrics.

In the inference process, independently from the VFL type you run (arbitered / honest), the inference process is run
on the master and member:

.. code-block:: python

    # If the master does not hold the VFL model
    master_no_model = master_class(
        ... # Here go the params specific to the master class
        inference_target_uids=inference_target_uids, # Simulation of the partial data availability on the agent
        eval_batch_size=config.vfl_model.eval_batch_size,
        do_predict=True, # config.vfl_model.do_predict
    )
    ...

    # Otherwise, it should be defined with the same arguments as the member:
    master_with_model = master_class(
        ... # Here go the params specific to the master class
        inference_target_uids=inference_target_uids, # Simulation of the partial data availability on the agent
        eval_batch_size=config.vfl_model.eval_batch_size,
        do_predict=True, # config.vfl_model.do_predict
        model_path=config.vfl_model.vfl_model_path,
    )
    ...
    # On the other hand, member always holds a model part
    member = member_class(
        ...
        member_inference_record_uids=inference_target_uids, # Simulation of the partial data availability on the agent
        eval_batch_size=config.vfl_model.eval_batch_size,
        do_predict=True, # config.vfl_model.do_predict
        model_path=config.vfl_model.vfl_model_path,
    )



Launching inference using the Stalactite CLI
------------------------------------------------

No matter how the configuration parameters `do_train` and `do_predict` are defined, in the Stalactite CLI, the
training or prediction is launched with respect to the passed command.

Local (single or multi process) prediction
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For the local inference (on the same computational node), the stalactite CLI  ``predict`` command group is used.
Depending on the flag (``--single-process`` | ``--multi-process``) either single Python process with multiple threads
or multiple processes in different containers.

To launch the inference, passing the configuration file path same to the training experiment, just type:

.. code-block:: bash

    stalactite predict [--multi-process] [--single-process] --config-path <path-to-defined/config.yml>

Distributed (multiple host) prediction
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Distributed inference uses the stalactite CLI  ``master`` and ``member`` groups.
After the training, again using the same to the training process configuration files, run:

.. code-block:: bash

    # To launch master on the master host
    stalactite master start --infer --config-path <path-to-defined/config.yml> [-d]


    # To launch members on the members hosts
    stalactite member start --infer --rank <member_rank> --config-path <path-to-defined/config.yml> [-d]

