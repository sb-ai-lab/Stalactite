from pathlib import Path
import os
import uuid

import pytest
import tenseal as ts
import torch

from stalactite.communications.arbiter_client_grpc import GRpcArbiterCommunicator
from stalactite.configs import VFLConfig, raise_path_not_exist
from stalactite.main import BASE_IMAGE_TAG, BASE_IMAGE_FILE

CONTAINER_NAME = 'test-arbiter'


@pytest.fixture(scope='module')
def arbiter_servicer(test_config_path, docker_client):
    config = VFLConfig.load_and_validate(test_config_path)
    containers = docker_client.containers(all=True, filters={'name': CONTAINER_NAME})
    if len(containers) == 0:
        _logs = docker_client.build(
            path=str(Path(os.path.abspath(__file__)).parent.parent.parent),
            tag=BASE_IMAGE_TAG,
            quiet=True,
            decode=True,
            nocache=False,
            dockerfile=os.path.join(Path(os.path.abspath(__file__)).parent.parent.parent, 'docker', BASE_IMAGE_FILE),
        )

        configs_path = os.path.dirname(os.path.abspath(test_config_path))

        raise_path_not_exist(configs_path)

        volumes = [f'{configs_path}']
        host_config = docker_client.create_host_config(
            binds=[f'{configs_path}:{configs_path}:rw'],
            port_bindings={int(config.grpc_arbiter.port): (int(config.grpc_arbiter.port),)}
        )
        arbiter_container = docker_client.create_container(
            image=BASE_IMAGE_TAG,
            command=[
                "python",
                "communications/grpc_utils/grpc_arbiter.py",
                "--config-path",
                f"{os.path.abspath(test_config_path)}"
            ],
            detach=True,
            ports=[int(config.grpc_arbiter.port)],
            environment={},
            volumes=volumes,
            host_config=host_config,
            name=CONTAINER_NAME,
        )
        docker_client.start(container=arbiter_container.get('Id'))


@pytest.fixture(scope='module')
def arbiter(test_config_path, arbiter_servicer):
    config = VFLConfig.load_and_validate(test_config_path)
    arbiter = GRpcArbiterCommunicator(
        arbiter_host=config.grpc_arbiter.container_host,
        arbiter_port=config.grpc_arbiter.port,
        grpc_operations_timeout=3000.,
        max_message_size=-1,
        master_id=str(uuid.uuid4()),  # TODO add to config
    )
    arbiter.initialize_arbiter()
    return arbiter


def _close_tensors(
        tensor1: torch.Tensor,
        tensor2: torch.Tensor,
        atol: float = 0.01,
        dtype: torch.dtype = torch.float64
) -> bool:
    return torch.allclose(tensor1.to(dtype), tensor2.to(dtype), atol=atol)


@pytest.mark.parametrize(
    "decrypted_data1,decrypted_data2,decrypt_func",
    [
        (
                torch.tensor([60, 66, 73, 81]),
                torch.tensor([40, 34, 27, 19]),
                ts.ckks_vector
        ),
        (
                torch.tensor([[-60, 66, -73, 81], [60, 66, 73, 81]]),
                torch.tensor([[60, -66, 73, -81], [60, 66, 73, 81]]),
                ts.ckks_tensor
        ),
        (
                torch.rand(4, 5, 6, dtype=torch.float64),
                torch.rand(4, 5, 6, dtype=torch.float64),
                ts.ckks_tensor
        ),
    ]
)
def test_arbiter_exchange(arbiter, decrypted_data1, decrypted_data2, decrypt_func):
    encrypted_data1 = decrypt_func(arbiter.public_key, decrypted_data1)
    encrypted_data2 = decrypt_func(arbiter.public_key, decrypted_data2)
    encrypted_data_sum = encrypted_data1 + encrypted_data2

    decrypted_data_sum = decrypted_data1 + decrypted_data2

    decrypted_data_sum_arbiter = arbiter.decrypt_data(encrypted_data_sum)
    decrypted_data_arbiter1 = arbiter.decrypt_data(encrypted_data1)
    decrypted_data_arbiter2 = arbiter.decrypt_data(encrypted_data2)

    assert _close_tensors(decrypted_data_sum_arbiter, decrypted_data_sum)
    assert _close_tensors(decrypted_data_arbiter1 + decrypted_data_arbiter2, decrypted_data_sum)
