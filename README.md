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

3. You can run tests or check out the examples to see whether everything is working (`experiments/README.md`)

[//]: # (TODO add examples)
# local examples
# stalactite CLI
# distributed launch