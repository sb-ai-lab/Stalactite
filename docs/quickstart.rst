Quickstart on Stalactite
======================================

Stalactite framework already implements several training algorithm in a federated learning fashion.
Here we provide examples of launching local and distributed experiments.

Requirements
--------------------------------------
To run experiments all of the VFL agent hosts should meet the following requirements:

* Python 3.9+
* Docker (for the distributed and local multiprocess VFL experiments) и docker-compose
* `Poetry <https://python-poetry.org/docs/#installing-with-pipx>`_

Installation
--------------------------------------
0. Check that your system has Docker and Poetry

.. code-block:: bash

    docker --version
    poetry --version

1. Prepare poetry configuration and install Stalactite:

.. code-block:: bash

    poetry config virtualenvs.in-project true # Create the virtualenv inside the project’s root directory.
    # You can configure poetry using official docs: https://python-poetry.org/docs/configuration/
    # If you use CPU, install torch and torchvision by running
    poetry run pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cpu
    # Otherwise, to use GPU:
    poetry run pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121
    poetry install # Install stalactite and dependencies
    poetry shell # Start a new shell and activate the virtual environment

2. Check if the Stalactite CLI is working by running:

.. code:: bash

    stalactite --help

If everything is ok, you should see the following output:

.. code-block:: bash

    # Usage: stalactite [OPTIONS] COMMAND [ARGS]...
    #
    #   Main stalactite CLI command group.
    #
    # Options:
    #   --help  Show this message and exit.
    #
    # Commands:
    #   local          Local experiments (multi-process / single process) mode...
    #   master         Distributed VFL master management command group.
    #   member         Distributed VFL member management command group.
    #   predict
    #   prerequisites  Prerequisites management command group.
    #   report         Experimental report command group.
    #   test           Local tests (multi-process / single process) mode...


Examples
--------------------------------------
In the `examples folder <https://github.com/sb-ai-lab/vfl-benchmark/tree/main/examples>`_ there are examples of how to
launch the VFL experiments locally (for the debug) or in a distributed fashion.
All the experiments utilize configuration files in YAML format to set the experimental parameters.

Configuration file
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
For the full description on the configuration file fields checkout: :ref:`config_tutorial`

Configuration file contains settings for all the parts of the experiment:

* ``common`` group contains general experimental settings;
* ``data`` part adjusts custom dataset into the experiment;
* ``prerequisites`` section manages the MlFlow anf Prometheus usage for logging and monitoring;
* ``grpc_server`` section used only in distributed experiments and configures the gRPC server used in communication between VFL agents;
* ``master`` settings allow tuning of the VFL master instance;
* ``member`` settings customizes VFL member instance;
* ``docker`` section is required for prerequisites and distributed launches and must be customized to the experiments host.

Example configuration files are shown in
`examples configs folder <https://github.com/sb-ai-lab/vfl-benchmark/tree/main/examples/configs>`_, each example is
linked to its config.


Local experiments
----------------------------------------------------------------------------

The following section contains local (single process multiple threads) experiments examples description.
These experiments are useful for the VFL algorithms development and debugging.

Linear regression on MNIST
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``examples/vfl/local/linreg_mnist_local.py`` launches the local linear regression example on MNIST dataset.
The YAML file for this experiment ``examples/configs/linreg-mnist-local.yml`` configures main common and data
parameters required for the launch. In this file you should customize the following fields:

.. code-block:: yaml

    common:
      # Here you must pass the path to folder where reports can be exported
      reports_export_folder: "../reports"

    data:
      # Here you must pass the path to folder containing the dataset
      host_path_data_dir: "../data/mnist_binary38_parts2"

    prerequisites:
      # Host and port of the MlFlow server (if enabled)
      mlflow_host: 0.0.0.0
      mlflow_port: "5000"

    master:
      # Whether to enable and use MlFlow to log metrics and parameters
      run_mlflow: True

After you fixed the paths and MlFlow, you can run the file from terminal / your IDE, or run

.. code-block:: bash

    stalactite local --single-process start --config-path examples/configs/linreg-mnist-local.yml


Linear regression on MNIST (sequential updates)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


``examples/vfl/local/linreg_mnist_seq_local.py`` launches the local linear regression example on MNIST dataset with
sequential updates on members. The configuration for this experiment is ``examples/configs/linreg-mnist-seq-local.yml``
is basically same to the `Linear regression on MNIST`_ example, except for the following:

.. code-block:: yaml

    common:
      is_consequently: True

Do not forget to pass all the paths and check the MlFlow server configuration.
Now, you can run the file from terminal / your IDE, or launch an experiment using the stalactite CLI:

.. code-block:: bash

    stalactite local --single-process start --config-path examples/configs/linreg-mnist-seq-local.yml


Logistic regression on SBOL and SMM
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


``examples/vfl/local/logreg_sbol_smm_local.py`` launching the multilabel classification with logistic regression on
SBOL and SMM datasets. The configuration for this experiment is ``examples/configs/logreg-sbol-smm-local.yml``. Again,
the configuration is pretty similar, and you should pass the paths to dataset and reports folder. However, to change the
experiment from linear regression on one data to logistic regression on another dataset, the following parameters are
altered:

.. code-block:: yaml

    common:
      vfl_model_name: logreg
      is_consequently: False
      use_class_weights: False

    data:
      dataset: 'sbol'
      use_smm: True
      train_split: "train_train" # Name of the train split
      test_split: "train_val" # Name of the test split
      features_key: "features_part_" # Features columns
      label_key: "labels" # Target column


Now, you can run the file from terminal / your IDE, or launch an experiment using the stalactite CLI:

.. code-block:: bash

    stalactite local --single-process start --config-path examples/configs/logreg-sbol-smm-local.yml


Distributed multiple process experiment
----------------------------------------------------------------------------

Here we will show you how to start a multi-process VFL experiment, in which each agent is a docker container on one host
machine.

Logistic regression on SBOL and SMM (MP)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We implemented helper shell script which demonstrates the usage of main Stalactite CLI commands for mutliple process
experiments (``examples/vfl/distributed/multiprocess/logreg_sbol_smm_multiprocess``)
The configuration file can be found at: ``examples/configs/logreg-sbol-smm-multiprocess.yml``

The main difference between distributed launch and `Logistic regression on SBOL and SMM`_ is the communicator. Instead
of LocalCommunicator we use gRPC server for master and member communications. Moreover, now we need to configure some
docker related parameters. Thus, in the configuration file we add new sections and fields (in comparison to the local
example):

.. code-block:: yaml

    prerequisites:
      # If we enanble logging to the Prometheus, we should introduce the host of the Prometheus container
      # Note, that VFL master and Prometheus must always be on the same host to see each other
      prometheus_host: '158.160.110.227'
      prometheus_port: '9090'
      grafana_port: '3001'

    grpc_server:
      # Those are default gRPC server settings, the gRPC server will be launched in the VFL master
      host: '0.0.0.0'
      port: '50051'
      # -1 means no limits are applied to the size of the send/recv message
      max_message_size: -1

    master:
      # Enable Prometheus if the prerequisites are running
      run_prometheus: True
      logging_level: 'debug'
      # gRPC communicator will consider the member disconnected if no pings were sent in `disconnect_idle_client_time`
      disconnect_idle_client_time: 120.

    member:
      logging_level: 'debug'
      # How often the member will send the heartbeats to the master
      heartbeat_interval: 2.

    docker:
      # Docker compose command on your machine ("docker compose" | "docker-compose")
      docker_compose_command: "docker compose"
      # Path to the docker-compose.yml file for the prerequisites (required for the Stalactite CLI prerequisites group)
      docker_compose_path: "../prerequisites"
      # Whether your machine uses GPU (required for correct torch dependencies in the containers)
      use_gpu: False

To launch the experiment run the following:

1. Stop previous launches by running the ``halt`` command

.. code-block:: bash

    bash examples/vfl/distributed/multiprocess/logreg_sbol_smm_multiprocess halt


2. Run the experiment

.. code-block:: bash

    bash examples/vfl/distributed/multiprocess/logreg_sbol_smm_multiprocess run


3. Check the logs of the master container

.. code-block:: bash

    bash examples/vfl/distributed/multiprocess/logreg_sbol_smm_multiprocess master-logs

4. You also can go to `http://<public-yc-ip>:5555/` to check the experiments metrics if the ``master.run_mlflow`` is set
to ``True`` in the config

Distributed multiple host experiment
----------------------------------------------------------------------------

Here we will show you how to start a multi-host VFL experiment, in which each agent is a docker container on several
host machines.

Logistic regression on SBOL and SMM (MH)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For each host the separate configuration file is required (due to possible differences in the host paths).
As the experiment example in launched across three virtual machines three postfixes identify which machine is used:
``yc`` for the YandexCloud, ``sber`` for the SberCloud, ``vk`` for the VK CLoud.
Configuration files for the MH experiment can be found at ``examples/configs/logreg-sbol-smm-vm-<postfix>.yml``.

Due to the master and prerequisites (if started) are launched on the same host, all the configs contain the same fields,
including:

.. code-block:: yaml

    prerequisites:
      mlflow_host: <master_host_public_ip>
      prometheus_host: <master_host_public_ip>

    master:
      container_host: <master_host_public_ip>

Nevertheless, the paths on different machines differ, therefore, paths to data and reports folders must be changed:

.. code-block:: yaml

    common:
      reports_export_folder: "../vfl-benchmark/reports"

    data:
      host_path_data_dir: "../vfl_multilabel_sber_sample10000_parts2"

After you configure the machines, you can use the helper script which launches master and members via ssh:

1. Stop previous launches by running the ``halt`` command

.. code-block:: bash

    bash examples/vfl/distributed/multihost/logreg_sbol_smm_multihost halt

2. Run the experiment

.. code-block:: bash

    bash examples/vfl/distributed/multihost/logreg_sbol_smm_multihost run

3. Check the logs of the master container

.. code-block:: bash

    bash examples/vfl/distributed/multihost/logreg_sbol_smm_multihost master-logs

4. You also can go to `http://<public-yc-ip>:5555/` to check the experiments metrics if the ``master.run_mlflow`` is set
to ``True`` in the config
