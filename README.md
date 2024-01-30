# Stalactite VFL
___

_Stalactite_ is the framework implementing the vertical federated learning paradigm. 

### Requirements
- Python 3.9+
- Docker (for the distributed and local multiprocess VFL experiments) и docker-compose
- [Poetry](https://python-poetry.org/docs/#installing-with-pipx) 

## Installation
0. Check that your system has Docker and Poetry
```bash
docker --version
poetry --version
```
1. Prepare poetry configuration and install Stalactite:
```bash
poetry config virtualenvs.in-project true # Create the virtualenv inside the project’s root directory.
# You can configure poetry using official docs: https://python-poetry.org/docs/configuration/
poetry install # Install stalactite and dependencies
poetry shell # Start a new shell and activate the virtual environment
```
2. Check if the Stalactite CLI is working by running:
```bash
stalactite --help

#Usage: stalactite [OPTIONS] COMMAND [ARGS]...
#
#  Main stalactite CLI command group.
#
#Options:
#  --help  Show this message and exit.
#
#Commands:
#  local          Local experiments (multi-process / single process) mode...
#  master         Distributed VFL master management command group.
#  member         Distributed VFL member management command group.
#  predict
#  prerequisites  Prerequisites management command group.
#  report         Experimental report command group.
#  test           Local tests (multi-process / single process) mode...
```

## Examples 
You can run tests or check out the examples to see whether everything is working (`examples/vfl`)
 In the examples folder there are examples of how to launch the VFL experiments locally (for the debug)
or in a distributed fashion.
For all the experiments to access the data, you should add the path to folder containing the
data into the config file field: ``

Local (single process multiple threads) experiments examples:
- `examples/vfl/local/linreg_mnist_local.py` launches the local linear regression example on MNIST dataset.
The YAML file for this experiment `examples/configs/local/linreg_mnist_local.yml` configures main common and data 
parameters required for the launch. 
- Same with the `examples/vfl/local/logreg_sbol_smm_local.py`, launching the multilabel classification with 
logistic regression on SBOL and SMM datasets.

Distributed (single host multiple processes (containers)) example:
- `examples/vfl/distributed/logreg_sbol_smm_multiprocess` example demonstrates the launch of the local multi-process 
stalactite CLI usage for running all the agents (master and members) containers on a single-host. 
In runs the same logistic regression example of SBOL and SMM. If you do not want 
to start prerequisites (or use them) while checking out the example, just disable the usage of the Prometheus or MlFLow by
changing configuration files fields to: `master.run_mlflow: False`, `master.run_prometheus: False`

Distributed (multiple host) example:
- `examples/vfl/dostributed/logreg_sbol_smm_distributed/` contains 3 shell scripts for launching VFL agents to run 
logistic regression on SBOL and SMM while all the agents are on different hosts. To launch it again uses the 
stalactite CLI and configuration file, which must be copied to each host (and changed accordingly).
  1. Adjust host addresses in the fields `master.container_host`, `prerequisites.mlflow_host`, 
  `prerequisites.prometheus_host`, or just the `master.container_host` if the prerequisites are disabled 
  (`master.run_mlflow: False`, `master.run_prometheus: False`).
  2. Fix data paths and run script `run_master` on the master 
  host (host, which address is passed into `master.container_host`). 
  3. Fix data paths and run script `run_member_0` and `run_member_1` on other hosts (or the same host) to start members.

## Prerequisites start and Stalactite CLI usage
To run the experiment and check the metrics, first, you should launch the prerequisites (Prometheus and MlFlow).
> **It is important to launch the prerequisites on the same host, where the distributed master will be running**

```bash
stalactite prerequisites start -d --config-path path/to/the/experiment-config.yml
# You can omit the `-d` option to attach the prerequisites to the current terminal
```

To run the experiments check the `stalactite/main` documentation or type `stalactite <command> --help`. You can also 
refer to the examples in the `Examples` section of the current README.

