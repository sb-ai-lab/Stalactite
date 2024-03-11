import mlflow
import numpy as np
import tenseal as ts
from phe import paillier

from arrays_phe import catchtime, ArrayEncryptor, ArrayDecryptor, matr_right_prod


def max_abs_diff(arr1: np.ndarray, arr2: np.ndarray):
    print(arr1[:2, :2])
    print(arr2[:2, :2])
    return np.max(np.abs(arr1 - arr2))

def shape_to_name(arr_shape: tuple[int, int]):
    return f"_{arr_shape[0]}x{arr_shape[1]}_"


class TensealSP:
    def __init__(self, poly_modulus_degree: int, coeff_mod_bit_sizes: list[int], global_scale_pow: int, n_threads: int):
        self.poly_modulus_degree = poly_modulus_degree
        self.coeff_mod_bit_sizes = coeff_mod_bit_sizes
        self.global_scale = 2 ** global_scale_pow
        self.n_threads = n_threads

        self.context = None

    def initialize(self):
        context = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=self.poly_modulus_degree,
            coeff_mod_bit_sizes=self.coeff_mod_bit_sizes,
            n_threads=self.n_threads,
        )
        context.generate_galois_keys()
        # context.generate_relin_keys()
        # context.global_scale = 2 ** 40
        context.global_scale = self.global_scale
        self.context = context

    def encrypt(self, array: np.ndarray, log: bool) -> ts.CKKSTensor:
        with catchtime() as time_encryption:
            arr_anc = ts.ckks_tensor(self.context, array)

        if log:
            mlflow.log_metric(f'time_encryption', time_encryption.time)
        return arr_anc

    def decrypt(self, arr_enc: ts.CKKSTensor, log: bool) -> np.ndarray:
        with catchtime() as time_decryption:
            array = np.array(arr_enc.decrypt().tolist())
        if log:
            mlflow.log_metric(f'time_decryption', time_decryption.time)
        return array

    def add_cypher_cypher(self, arr_enc1: ts.CKKSTensor, arr_enc2: ts.CKKSTensor) -> ts.CKKSTensor:
        with catchtime() as time_addition:
            addition = arr_enc1 + arr_enc2
        mlflow.log_metric(f'time_addition', time_addition.time)
        return addition

    def multiply_plain_cypher(self, plain: np.ndarray, cypher: ts.CKKSTensor) -> ts.CKKSTensor:
        # plain_enc = ts.ckks_tensor(self.context, plain)
        desired_shape = (plain.shape[0], cypher.shape[1])

        with catchtime() as time_multiplication:
            # mult = plain_enc.dot(cypher)
            mult = cypher.transpose().mm(plain.T).transpose()

        assert tuple(mult.shape) == desired_shape, f'Something went wrong, {mult.shape}, {desired_shape}'
        mlflow.log_metric(f'time_multiplication_{plain.shape[0]}', time_multiplication.time)
        return mult


class PailierSP:
    def __init__(self, precision: float, encoding_precision: float, n_jobs: int, key_n_length: int = 512):
        self.precision = precision
        self.encoding_precision = encoding_precision
        self.n_jobs = n_jobs
        self.key_n_length = key_n_length

        self.array_encryptor = None
        self.array_decryptor = None

    def initialize(self):
        public_key, private_key = paillier.generate_paillier_keypair(n_length=self.key_n_length)
        array_encryptor = ArrayEncryptor(public_key, n_jobs=self.n_jobs, precision=self.precision, encoding_precision=self.encoding_precision)
        array_decryptor = ArrayDecryptor(private_key, n_jobs=self.n_jobs)

        self.array_encryptor = array_encryptor
        self.array_decryptor = array_decryptor

    def encrypt(self, array: np.ndarray, log: bool) -> np.ndarray:
        with catchtime() as time_encryption:
            arr_anc = self.array_encryptor.encrypt(array)

        if log:
            mlflow.log_metric(f'time_encryption', time_encryption.time)
        return arr_anc

    def decrypt(self, arr_enc: np.ndarray, log: bool) -> np.ndarray:
        with catchtime() as time_decryption:
            array = self.array_decryptor.decrypt(arr_enc)
        if log:
            mlflow.log_metric(f'time_decryption', time_decryption.time)
        return array

    def add_cypher_cypher(self, arr_enc1: np.ndarray, arr_enc2: np.ndarray) -> np.ndarray:
        with catchtime() as time_addition:
            addition = arr_enc1 + arr_enc2
        mlflow.log_metric(f'time_addition', time_addition.time)
        return addition

    def multiply_plain_cypher(self, plain: np.ndarray, cypher: np.ndarray) -> np.ndarray:
        desired_shape = (plain.shape[0], cypher.shape[1])
        print('Encoding array for multiplication:', self.encoding_precision, plain.shape)
        plain_encoded = self.array_encryptor.encode(plain)

        with catchtime() as time_multiplication:
            mult = matr_right_prod(plain_encoded, cypher, n_jobs=self.n_jobs)

        assert mult.shape == desired_shape, f'Something went wrong, {mult.shape}, {desired_shape}'
        mlflow.log_metric(f'time_multiplication_{plain.shape[0]}', time_multiplication.time)
        return mult
