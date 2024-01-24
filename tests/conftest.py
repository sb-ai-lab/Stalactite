from docker import APIClient
import pytest


def pytest_addoption(parser):
    parser.addoption("--test_config_path", action="store", default="configs/config-test.yml")


@pytest.fixture(scope='session')
def test_config_path(request):
    test_config_path = request.config.option.test_config_path
    if test_config_path is None:
        pytest.skip()
    return test_config_path


@pytest.fixture(scope='session')
def docker_client() -> APIClient:
    client = APIClient(base_url='unix://var/run/docker.sock')
    return client
