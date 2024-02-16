from typing import Any

import torch
from phe import paillier

from stalactite.ml.arbitered.base import SecurityProtocolArbiter, SecurityProtocol, Keys


class SecurityProtocolPaillier(SecurityProtocol):

    def encrypt(self, data: torch.Tensor) -> Any:
        # TODO
        return data

    def drop_private_key(self) -> Keys:
        return Keys(private=None, public=self._keys.public)


class SecurityProtocolArbiterPaillier(SecurityProtocolPaillier, SecurityProtocolArbiter):

    def decrypt(self, decrypted_data: Any) -> torch.Tensor:
        # TODO
        return decrypted_data

    def generate_keys(self) -> None:
        public_key, private_key = paillier.generate_paillier_keypair()
        self._keys = Keys(public=public_key, private=private_key)
