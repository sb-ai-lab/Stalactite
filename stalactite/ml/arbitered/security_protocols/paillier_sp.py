from typing import Any

import torch
from phe import paillier

from stalactite.ml.arbitered.base import SecurityProtocolArbiter, SecurityProtocol, Keys


class SecurityProtocolArbiterPaillier(SecurityProtocolArbiter):
    def drop_private_key(self) -> Keys:
        return Keys(private=None, public=self._keys.public)

    def encrypt(self, data: torch.Tensor) -> Any:
        pass

    def decrypt(self, decrypted_data: Any) -> torch.Tensor:
        ...

    def generate_keys(self) -> None:
        public_key, private_key = paillier.generate_paillier_keypair()
        self._keys = Keys(public=public_key, private=private_key)
