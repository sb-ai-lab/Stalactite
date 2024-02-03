.. _config_tutorial:

Configuration files
======================================

The configuration files are used for tuning the experimental scripts to your needs.
An example of configuration file can be found at the ``configs/config-test.yml``.

Common parameters are required in any experiment and define general experimental specifics.

.. code-block:: yaml

    common:
      epochs: # Number of training epochs
      report_train_metrics_iteration: # Number of iteration steps between reporting metrics on train dataset split
      report_test_metrics_iteration: # Number of iteration steps between reporting metrics on test dataset split
      world_size: # Number of members in an experiment
      batch_size: # Training batch size
      experiment_label: # Label of the experiment
      reports_export_folder: # Path to export tests logs and reports
      vfl_model_name: # Model name to train
      # For local experiment with `linreg` model you can choose the consequent batcher, to update on member at a time
      is_consequently: False # Set True for consequent batcher
      learning_rate: # Experiment learning rate
      use_class_weights: # Used in `logreg`
      rendezvous_timeout: # If master and members do not finish rendezvous in a given time, TimeoutError is raised

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

.. code-block:: yaml

    master:
      container_host: # Host of the master container, which can be accessed by the members
      disconnect_idle_client_time: # Master will disconnect a member which has not sent any pings for `disconnect_idle_client_time`
      time_between_idle_connections_checks: # How often master should check disconnected members

    member:
      heartbeat_interval: # Interval of the heartbeat messages sent to master

    grpc_server:
      host: 0.0.0.0 # Which host is used inside the container to launch the gRPC server
      port: # Which port is used to launch and access gRPC server
      max_message_size: -1 # Maximum message size in bytes, -1 means no limits are applied
      server_threadpool_max_workers: # When running the gRPC servicer the threadpool workers are used

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
      docker_compose_path: # Path to the docker-compose.yml file and prerequisites configs/

