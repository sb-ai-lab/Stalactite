.. _config_tutorial:

*how-to:* Write configuration files
======================================

The configuration files are used for tuning the experimental scripts to your needs.
An example of configuration file can be found at the ``configs/config-test.yml``.

All the path parameters (``reports_export_folder``, ``host_path_data_dir``, ``docker_compose_path``) should be either
absolute paths to the directories, or are resolved relatively to the path of the configuration file.

The logging in the module is also defined via configuration files. We use standard Python logger and you can set the
logging level to ``info``, ``debug`` or ``warning`` using the following instruction.

.. code-block:: yaml

    # general logging level of the Stalactite, applied if you use Stalactite CLI in local or distributed experiments
    common:
      logging_level: # 'info', 'debug' or 'warning'

    # Agent level logging is applicable in distributed multi-process and multi-host regimes - you can set different
    # levels to different agents by customizing the following fields:
    master:
      logging_level: # 'info', 'debug' or 'warning': Logging level of the master (in master container)
    member:
      logging_level: # 'info', 'debug' or 'warning': Logging level of the member (in member container)
    grpc_arbiter:
      logging_level: # 'info', 'debug' or 'warning': Logging level of the arbiter (in arbiter container)


Common parameters are required in any experiment and define general experimental specifics.

.. note::
    Note, that ``common.world_size`` parameter identifies number of **member** agents in an experiment. So, if you run
    an *honest* experiment, number of agents (threads in *local* or containers in *distributed* case) will be
    ``common.world_size + 1``: ``common.world_size`` members and 1 master. In case of an ``arbitered`` experiment,
    number of agents will be ``common.world_size + 2``, where 1 is launched for the master, and 1 more - for an arbiter.

.. code-block:: yaml

    common:
      report_train_metrics_iteration: # Number of iteration steps between reporting metrics on train dataset split
      report_test_metrics_iteration: # Number of iteration steps between reporting metrics on test dataset split
      world_size: # Number of members in an experiment
      experiment_label: # Label of the experiment
      reports_export_folder: # Path to export tests logs and reports
      rendezvous_timeout: # If master and members do not finish rendezvous in a given time, TimeoutError is raised
      logging_level: # 'info', 'debug' or 'warning' - general logging level of the Stalactite

VFL model are training and model specific parameters also used in any experiment.

.. code-block:: yaml

    vfl_model:
      vfl_model_name: # Model name to train
      vfl_model_path: # Directory to save the model for further evaluation
      do_train: # Whether to do training of the model
      do_predict: # Whether to do evaluation of the model
      do_save_model: # Whether to save the model to the `vfl_model_path` after training
      epochs: # Number of training epochs
      batch_size: # Training batch size
      eval_batch_size: # Evaluation batch size
      # For local experiment with `linreg` model you can choose the consequent make_batcher, to update on member at a time
      is_consequently: False # Set True for consequent make_batcher
      learning_rate: # Experiment learning rate
      use_class_weights: # Used in `logreg`


Same applies to data parameters, which are required in each experiment and define main specifics of the dataset
used for training and validation.

.. code-block:: yaml

    data:
      random_seed: # Random seed for the experiment reproduction
      dataset_size: # Number of dataset rows to use
      host_path_data_dir: # Path to directory containing datasets
      dataset: # Dataset name (one of `mnist`, `sbol`)
      use_smm: # If using `sbol` dataset defines whether to add `smm` data
      dataset_part_prefix: # Used in dataset folder structure inspection. Concatenated with the index of a party: 0,1,... etc.
      train_split: # Name of the train split
      test_split: # Name of the test split
      features_key: # Feature columns key
      label_key: # Target column key
      num_classes: # Number of classes in the multiclass classification task (used only in arbitered OVR setting)

Master and member configuration fields can be split into two main groups. Required parameters for both local and
distributed experiments are the following.

.. code-block:: yaml

    master:
      run_mlflow: # If the prerequisites are launched defines whether to report metrics and parameters to MLFlow
      run_prometheus: # If the prerequisites are launched defines whether to report metrics to Prometheus
      logging_level: # Logging level of the master
      recv_timeout: # Timeout of the recv (and gather) operations on master

    member:
      logging_level: # Logging level of the member
      recv_timeout: # Timeout of the recv operations on member

Rest of the parameters are used only in the distributed setting.

In the arbitered setting, you need to configure an arbiter agent, too. Usage of the arbiter implies the introduction of
the homomorphic encryption in the process.

.. code-block:: yaml

    grpc_arbiter:
      use_arbiter: True # To launch an arbiter in an experiment
      logging_level: # Logging level of the arbiter

      # You can scip initialization of the `security_protocol_params`, no HE will be added into training
      security_protocol_params:
        he_type: paillier # By now only paillier HE is available
        # Lower key length means faster operations, worse precision and security. In a real-world setting, we recommend
        # setting the `key_length` to 2048
        key_length: 128
        n_threads: 20 # Number of available for parallelization CPU cores
        encryption_precision: 1e-10 # Precision of the encryption
        # (if the overflow error occurs, reduce this value or increase the key length)
        encoding_precision: 1e-10 # Precision of the encoding
        # (if the overflow error occurs, reduce this value or increase the key length)
      recv_timeout: # Timeout of the recv operations on arbiter


.. code-block:: yaml

    master:
      external_host: # Host of the master container, which can be accessed by the members
      disconnect_idle_client_time: # Master will disconnect a member which has not sent any pings for `disconnect_idle_client_time`
      time_between_idle_connections_checks: # How often master should check disconnected members

    member:
      heartbeat_interval: # Interval of the heartbeat messages sent to master

    grpc_server:
      port: # Which port is used to launch and access gRPC server
      max_message_size: -1 # Maximum message size in bytes, -1 means no limits are applied
      server_threadpool_max_workers: # When running the gRPC servicer the threadpool workers are used

    grpc_arbiter:
      port: # Which port is used to launch and access gRPC server !must be different from ``grpc_server.port``
      external_host: # Host of the arbiter container, which can be accessed by the members and master

    docker:
      # When containers are launched the built image depends on whether the GPU is available
      # Image without the GPU is significantly lighter, thus you can disable the usage in order to save memory
      use_gpu: # Whether to use torch built for the GPU training and inference

Prerequisites parameters are needed if you want to use MlFlow and Prometheus for logging and metrics reporting

.. code-block:: yaml

    prerequisites:
      mlflow_host: # Host of the MlFlow server
      mlflow_port: '5000' # Port of the MlFlow server
      prometheus_host: # Host of the Prometheus, !must be at the same host as master
      prometheus_port: '9090' # Port of the Prometheus
      grafana_port: '3001' # Port of the Grafana

The host machine of the MlFlow, Prometheus and VFL master will use the ``docker`` parameters for managing containers
with the prerequisites

.. code-block:: yaml

    docker:
      docker_compose_command: # Docker compose command
      # Path to the docker-compose.yml file and prerequisites configs/
      docker_compose_path: "../prerequisites" # The default path is relative to the repo root

