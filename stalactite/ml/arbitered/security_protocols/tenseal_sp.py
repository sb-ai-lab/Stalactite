import logging
from functools import partial
from typing import Any, Union, Optional

import numpy as np
import torch
from joblib import Parallel, delayed
import tenseal as ts

from stalactite.ml.arbitered.base import SecurityProtocolArbiter, SecurityProtocol, Keys, T
from stalactite.helpers import log_timing


class SecurityProtocolTenseal(SecurityProtocol):
    def __init__(
            self,
            poly_modulus_degree: int,
            coeff_mod_bit_sizes: list[int],
            global_scale_pow: int,
            n_threads: Optional[int] = None
    ):
        self.poly_modulus_degree = poly_modulus_degree
        self.coeff_mod_bit_sizes = coeff_mod_bit_sizes
        self.global_scale = 2 ** global_scale_pow
        self.n_threads = n_threads

        self.is_initialized = False

    def initialize(self):
        self.is_initialized = True

    def encrypt(self, data: torch.Tensor) -> ts.CKKSTensor:
        if not self.is_initialized:
            raise RuntimeError(
                'Security protocol was not initialized. You should call the SecurityProtocolTenseal.initialize '
                'method before trying to encrypt the data.'
            )
        return ts.ckks_tensor(self.keys.public, data)

    def drop_private_key(self) -> Keys:
        return Keys(private=None, public=self.keys.public)

    def multiply_plain_cypher(self, plain_arr: Any, cypher_arr: T) -> T:
        assert plain_arr.shape[-1] == cypher_arr.shape[0], "Arrays' shapes must be suitable for matrix " \
                                                           f"product {plain_arr.shape}, {cypher_arr}"

        return cypher_arr.transpose().mm(plain_arr.T).transpose()

    def add_matrices(self, array1: Union[torch.Tensor, ts.CKKSTensor], array2: ts.CKKSTensor) -> ts.CKKSTensor:
        return array1 + array2


class SecurityProtocolArbiterTenseal(SecurityProtocolTenseal, SecurityProtocolArbiter):
    def __init__(
            self,
            poly_modulus_degree: int,
            coeff_mod_bit_sizes: list[int],
            global_scale_pow: int,
            n_threads: Optional[int] = None
    ):
        super().__init__(poly_modulus_degree, coeff_mod_bit_sizes, global_scale_pow, n_threads)
        self.poly_modulus_degree = poly_modulus_degree
        self.coeff_mod_bit_sizes = coeff_mod_bit_sizes
        self.global_scale = 2 ** global_scale_pow
        self.n_threads = n_threads

    def generate_keys(self):
        context = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=self.poly_modulus_degree,
            coeff_mod_bit_sizes=self.coeff_mod_bit_sizes,
            n_threads=self.n_threads,
        )
        context.generate_galois_keys()
        context.generate_relin_keys()
        context.global_scale = self.global_scale

        context_pub = context.copy()
        context_pub.make_context_public()

        self._keys = Keys(public=context_pub, private=context)
        self.initialize()

    def decrypt(self, encrypted_data: Union[ts.CKKSTensor, bytes]) -> torch.Tensor:
        if isinstance(encrypted_data, bytes):
            encrypted_data = ts.ckks_tensor_from(self.keys.private, encrypted_data)

        data = encrypted_data.decrypt()
        return torch.tensor(data.raw).reshape(data.shape)
