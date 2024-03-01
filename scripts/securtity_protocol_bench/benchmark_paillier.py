import mlflow
import click
import numpy as np

from helpers import PailierSP, max_abs_diff, shape_to_name
from constants import ARR_SIZES, PLAIN_ARR_SIZE


@click.command
@click.option(
    "--mlflow-addr", type=str, required=False, help="MlFlow URI.", default='node3.bdcl:5555',
)
@click.option(
    "-e", "--experiment", type=str, required=False, help="MlFlow experiment name.", default="benchmark-paillier"
)
@click.option("--key-n-length", type=int, required=False, help="Length of the generated key.", default=512)
@click.option("--round", type=int, required=False, help="Round numpy arrays.", default=32)
def main(mlflow_addr: str, experiment: str, key_n_length: int, round: int):
    # 'node3.bdcl:5555'
    mlflow.set_tracking_uri(f"http://{mlflow_addr}")
    experiment = f"{experiment}-{key_n_length}"
    mlflow.set_experiment(experiment)

    threads = [20, 40]
    precisions = [1e-8, 1e-15]

    for n_threads in threads:
        for precision in precisions:
            for array_size in ARR_SIZES:
                with mlflow.start_run():
                    log_params = {
                        "n_jobs": n_threads,
                        "precision": precision,
                        "array_size": shape_to_name(array_size),
                        "round": round,
                        "key_n_length": key_n_length,
                    }
                    mlflow.log_params(log_params)

                    paillier_sp = PailierSP(
                        precision=precision,
                        n_jobs=n_threads,
                        key_n_length=key_n_length
                    )
                    paillier_sp.initialize()

                    print('Generating arrays')

                    s1 = np.random.uniform(-1e+5, 1e+5, size=array_size)
                    s2 = np.random.uniform(-1e+5, 1e+5, size=array_size)

                    mant, expn = np.frexp(s1)
                    s1 = np.ldexp(np.round(mant, round), expn)

                    mant, expn = np.frexp(s2)
                    s2 = np.ldexp(np.round(mant, round), expn)

                    print('Encrypting arrays')
                    s1_enc = paillier_sp.encrypt(s1, log=True)
                    s2_enc = paillier_sp.encrypt(s2, log=False)

                    print('Addition')
                    sum_enc_s1s2 = paillier_sp.add_cypher_cypher(s1_enc, s2_enc)

                    print('Decrypting arrays')
                    sum_enc_s1s2_decr = paillier_sp.decrypt(sum_enc_s1s2, log=False)
                    s1_enc_decr = paillier_sp.decrypt(s1_enc, log=True)
                    s2_enc_decr = paillier_sp.decrypt(s2_enc, log=False)

                    mlflow.log_metric(
                        'all_close_decr_sum',
                        np.allclose(sum_enc_s1s2_decr, s1 + s2, rtol=1e-05, atol=1e-08, equal_nan=False)
                    )
                    mlflow.log_metric(
                        'max_abs_diff_decr_sum',
                        max_abs_diff(sum_enc_s1s2_decr, s1 + s2)
                    )

                    mlflow.log_metric(
                        'all_close_sum_decr',
                        np.allclose(s1_enc_decr + s2_enc_decr, s1 + s2, rtol=1e-05, atol=1e-08, equal_nan=False)
                    )
                    mlflow.log_metric(
                        'max_abs_diff_sum_decr',
                        max_abs_diff(s1_enc_decr + s2_enc_decr, s1 + s2)
                    )

                    for plain_size in PLAIN_ARR_SIZE:
                        plain_arr = np.random.uniform(-1e-1, 1e-1, size=(plain_size, array_size[0]))

                        mant, expn = np.frexp(plain_arr)
                        plain_arr = np.ldexp(np.round(mant, round), expn)

                        print('Multiplying')
                        mult_enc_s1s2 = paillier_sp.multiply_plain_cypher(plain_arr, s1_enc)
                        print('Decrypting arrays')
                        mult_enc_s1_decr = paillier_sp.decrypt(mult_enc_s1s2, log=False)

                        mult_s1 = np.dot(plain_arr, s1)

                        mlflow.log_metric(
                            f'close_decr_mult_{plain_size}',
                            np.allclose(mult_s1, mult_enc_s1_decr, rtol=1e-05, atol=1e-08, equal_nan=False)
                        )

                        mlflow.log_metric(
                            f'max_abs_diff_mult_{plain_size}',
                            max_abs_diff(mult_s1, mult_enc_s1_decr)
                        )


if __name__ == '__main__':
    main()
