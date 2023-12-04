from stalactite.base import DataTensor


class PrivacyGuard:
    method: str  # or strategy
    budget: float

    def __init__(self, method):
        assert method in ['DP', 'HE']
        self.method = method

    def add_gaussian_noise(self, sigma: float, shape) -> DataTensor:
        ...

    def norm_clipping(self, input: DataTensor, k: float) -> DataTensor:
        ...

    def norm_penalty(self) -> DataTensor:
        ...

    def set_public_key(self) -> DataTensor:
        ...

    def encrypt(self) -> DataTensor:
        ...

    def decrypt(self) -> DataTensor:
        ...
