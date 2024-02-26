import logging
from functools import partial
from typing import Any, Union

import numpy as np
import torch
from joblib import Parallel, delayed
from phe import paillier

from stalactite.ml.arbitered.base import SecurityProtocolArbiter, SecurityProtocol, Keys
from stalactite.helpers import log_timing


# from arrays_phe import ArrayEncryptor, ArrayDecryptor, encrypted_array_size_bytes, catchtime, pretty_print_params


class SecurityProtocolPaillier(SecurityProtocol):
    def multiply_plain_cypher(self, plain_arr: np.ndarray, cypher_arr: np.ndarray) -> np.ndarray:
        assert plain_arr.shape[-1] == cypher_arr.shape[0], "Arrays' shapes must be suitable for matrix " \
                                                           f"product {plain_arr.shape}, {cypher_arr}"

        if self.n_jobs == 0:
            res = np.dot(plain_arr, cypher_arr)
        else:
            n_jobs_eff = min((plain_arr.shape[0] // 3) + 1, self.n_jobs)
            if n_jobs_eff == 0:
                return np.dot(plain_arr, cypher_arr)

            with Parallel(self.n_jobs) as p:
                res = np.concatenate(
                    p(delayed(np.dot)(x, cypher_arr) for x in np.array_split(plain_arr, n_jobs_eff))
                )
        return res

    def add_matrices(self, array1: np.ndarray, array2: np.ndarray) -> np.ndarray:
        if self.n_jobs == 0:
            res = array1 + array2
        else:
            n_jobs_eff = min((array1.shape[0] // 3) + 1, self.n_jobs)
            if n_jobs_eff == 0:
                return array1 + array2

            assert array1.shape == array2.shape, "Arrays' shapes must be equal for encrypted " \
                                                 f"addition {array1.shape}, {array2.shape}"
            orig_shape = array1.shape

            add = lambda x1, x2: x1 + x2
            with Parallel(self.n_jobs) as p:
                res = np.concatenate(
                    p(
                        delayed(add)(x, y) for x, y in zip(
                            np.array_split(array1.reshape(-1), n_jobs_eff),
                            np.array_split(array2.reshape(-1), n_jobs_eff)
                        )
                    )
                )
            res = res.reshape(orig_shape)
        return res

    def __init__(self, precision: float = 1e-10, n_threads: int = 3, ):
        self.precision = precision
        self.n_jobs = n_threads

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

            if (np_data.size < 10) or (self.n_jobs == 0) or (self.n_jobs == 1) or (n_jobs_eff == 0) or (
                    n_jobs_eff == 1):
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
    def __init__(self, precision: float = 1e-10, n_threads: int = 3, ):
        super().__init__(precision=precision, n_threads=n_threads)

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

    def decrypt(self, encrypted_data: Any) -> torch.Tensor:
        with log_timing(f'Decrypting the data of len: {len(encrypted_data)}', log_func=print):
            if not self.is_initialized:
                raise RuntimeError(
                    'Security protocol was not initialized. You should generate keys '
                    'before trying to decrypt the data.'
                )
            data_np = np.array(encrypted_data)
            n_jobs_eff = min((data_np.size // 3) + 1, self.n_jobs)

            if (data_np.size < 10) or (self.n_jobs == 0) or (self.n_jobs == 1) or (n_jobs_eff == 0) or (
                    n_jobs_eff == 1):
                return self.vec_decrypt(data_np)
            else:
                orig_shape = data_np.shape
                with Parallel(n_jobs_eff) as p:
                    res = np.concatenate(
                        p(delayed(self.vec_decrypt)(x) for x in np.array_split(data_np.reshape(-1), n_jobs_eff))
                    )
                res = res.reshape(orig_shape)
                return torch.from_numpy(res)

    def generate_keys(self) -> None:
        public_key, private_key = paillier.generate_paillier_keypair()
        self._keys = Keys(public=public_key, private=private_key)
        self.initialize()
