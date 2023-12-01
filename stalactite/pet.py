from stalactite.base import DataTensor


class PrivacyGuard:
    method: str  # or strategy
    budget: float

    def __init__(self, method):
        assert method in ['DP', 'MPC', 'HE', 'TEE', 'GD']
        self.method = method

    def gaussian_noise(self, sigma: float, shape):
        ...

    def norm_clipping(self, input: DataTensor, k: float) -> DataTensor:
        ...

    def norm_penalty(self):
        ...

    def set_public_key(self) -> DataTensor:
        ...

    def encrypt(self) -> DataTensor:
        ...

    def decrypt(self) -> DataTensor:
        ...
