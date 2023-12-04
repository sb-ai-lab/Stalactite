from stalactite.base import DataTensor
from abc import ABC, abstractmethod


class PrivacyGuard:
    def __init__(self, method: str):
        self.method = str  # or strategy
        self._initialize()

    def _initialize(self):
        if self.method == 'HE':
            return PrivacyGuardHE()
        if self.method == 'DP':
            return PrivacyGuardDP()

    def add_gaussian_noise(self, sigma: float, shape) -> DataTensor:
        ...

    def norm_clipping(self, input: DataTensor, k: float) -> DataTensor:
        ...

    def norm_penalty(self) -> DataTensor:
        ...

    def send_public_key(self) -> DataTensor:
        ...

    def encrypt(self) -> DataTensor:
        ...

    def decrypt(self) -> DataTensor:
        ...


class PrivacyGuardHE(PrivacyGuard):
    def __init__(self):
        ...


class PrivacyGuardDP(PrivacyGuard):
    def __init__(self):
        ...
