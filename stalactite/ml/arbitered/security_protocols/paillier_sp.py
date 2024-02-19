import logging
from functools import partial
from typing import Any

import numpy as np
import torch
from joblib import Parallel, delayed
from phe import paillier

from stalactite.ml.arbitered.base import SecurityProtocolArbiter, SecurityProtocol, Keys
from stalactite.helpers import log_timing

# from arrays_phe import ArrayEncryptor, ArrayDecryptor, encrypted_array_size_bytes, catchtime, pretty_print_params


class SecurityProtocolPaillier(SecurityProtocol):
    def __init__(self, precision: float = 1e-10, n_jobs: int = 3, ):
        self.precision = precision
        self.n_jobs = n_jobs

        self.is_initialized = False
        self.enc_partial_function = None
        self.enc_encoded_partial_function = None
        self.vec_encrypt = None
        self.vec_encrypt_encoded = None

    def initialize(self):
        self.enc_partial_function = partial(self._keys.public.encrypt, precision=self.precision)
        self.enc_encoded_partial_function = lambda x: self._keys.public.encrypt_encoded(x, 1)
        self.vec_encrypt = np.vectorize(
            self.enc_partial_function,
            otypes=None,
            doc=None,
            excluded=None,
            cache=False,
            signature=None
        )

        self.vec_encrypt_encoded = np.vectorize(
            self.enc_encoded_partial_function,
            otypes=None,
            doc=None,
            excluded=None,
            cache=False,
            signature=None
        )
        self.is_initialized = True

    def encrypt(self, data: torch.Tensor) -> Any:
        with log_timing(f'Encryptying the tensor of shape: {data.shape}', log_func=print):
            if not self.is_initialized:
                raise RuntimeError(
                    'Security protocol was not initialized. You should call the SecurityProtocolPaillier.initialize '
                    'method before trying to encrypt the data.'
                )
            np_data = data.numpy(force=True).astype('float')
            n_jobs_eff = min((np_data.size // 3) + 1, self.n_jobs)

            if (np_data.size < 10) or (self.n_jobs == 0) or (self.n_jobs == 1) or (n_jobs_eff == 0) or (n_jobs_eff == 1):
                result = self.vec_encrypt(np_data)
            else:
                orig_shape = np_data.shape

                with Parallel(n_jobs_eff) as p:
                    res = np.concatenate(
                        p(delayed(self.vec_encrypt)(x) for x in np.array_split(np_data.reshape(-1), n_jobs_eff))
                    )

                res = res.reshape(orig_shape)

                result = res
            return result

    def drop_private_key(self) -> Keys:
        return Keys(private=None, public=self._keys.public)


class SecurityProtocolArbiterPaillier(SecurityProtocolPaillier, SecurityProtocolArbiter):
    def __init__(self, precision: float = 1e-10, n_jobs: int = 3, ):
        super().__init__(precision=precision, n_jobs=n_jobs)

        self.vec_decrypt = None
        self.vec_decrypt_encoded = None

    def initialize(self):
        super().initialize()
        self.vec_decrypt = np.vectorize(
            self._keys.private.decrypt,
            otypes=None,
            doc=None,
            excluded=None,
            cache=False,
            signature=None
        )

        self.vec_decrypt_encoded = np.vectorize(
            self._keys.private.decrypt_encoded,
            otypes=None,
            doc=None,
            excluded=None,
            cache=False,
            signature=None
        )

        self.is_initialized = True

    def decrypt(self, decrypted_data: Any) -> torch.Tensor:
        with log_timing(f'Decrypting the data of len: {len(decrypted_data)}', log_func=print):
            if not self.is_initialized:
                raise RuntimeError(
                    'Security protocol was not initialized. You should generate keys '
                    'before trying to decrypt the data.'
                )
            data_np = np.array(decrypted_data)
            n_jobs_eff = min((data_np.size // 3) + 1, self.n_jobs)

            if (data_np.size < 10) or (self.n_jobs == 0) or (self.n_jobs == 1) or (n_jobs_eff == 0) or (n_jobs_eff == 1):
                return self.vec_decrypt(data_np)
            else:
                orig_shape = data_np.shape
                with Parallel(n_jobs_eff) as p:
                    res = np.concatenate(
                        p(delayed(self.vec_decrypt)(x) for x in np.array_split(data_np.reshape(-1), n_jobs_eff))
                    )
                res = res.reshape(orig_shape)
                return res


    def generate_keys(self) -> None:
        public_key, private_key = paillier.generate_paillier_keypair()
        self._keys = Keys(public=public_key, private=private_key)
        self.initialize()
