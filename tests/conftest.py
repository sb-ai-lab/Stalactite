import os
import pytest
import pathlib


@pytest.fixture(scope='session')
def docker_compose_file():
    return os.path.join(pathlib.Path(os.path.abspath(__file__)).parent.parent, 'prerequisites', 'docker-compose.yml')


@pytest.fixture(scope='session')
def docker_setup():
    return ""


@pytest.fixture(scope='session')
def docker_cleanup():
    return ""
