import mlflow
import click
import numpy as np

from helpers import TensealSP, max_abs_diff, shape_to_name
from constants import ARR_SIZES, PLAIN_ARR_SIZE


@click.command
@click.option(
    "--mlflow-addr", type=str, required=False, help="MlFlow URI.", default='node3.bdcl:5555',
)
@click.option(
    "-e", "--experiment", type=str, required=False, help="MlFlow experiment name.", default="benchmark-tenseal"
)
def main(mlflow_addr: str, experiment: str):
    # 'node3.bdcl:5555'
    mlflow.set_tracking_uri(f"http://{mlflow_addr}")
    mlflow.set_experiment(experiment)

    threads = [20, 40]
    # scale = [20, 40]

    for n_threads in threads:
        for (poly_mod, coeff_mod_bit_sizes, scale_pow) in [
            (8192, [60, 40, 40, 60], 40),
            (8192 * 2, [60, 40, 40, 60], 40),
            (8192, [40, 21, 21, 21, 21, 21, 21, 40], 21),
            (8192, [40, 20, 40], 20),
            (4096, [40, 20, 40], 20),
            (4096, [30, 20, 30], 20),
        ]:
            for array_size in ARR_SIZES:

                with mlflow.start_run():
                    log_params = {
                        "poly_modulus_degree": poly_mod,
                        "coeff_mod_bit_sizes": coeff_mod_bit_sizes,
                        "scale_pow": scale_pow,
                        "n_threads": n_threads,
                        "array_size": shape_to_name(array_size),
                    }
                    mlflow.log_params(log_params)

                    tenseal_sp = TensealSP(
                        poly_modulus_degree=poly_mod,
                        coeff_mod_bit_sizes=coeff_mod_bit_sizes,
                        n_threads=n_threads,
                        global_scale_pow=scale_pow
                    )
                    tenseal_sp.initialize()
                    print('Generating arrays')

                    s1 = np.random.uniform(-1e+5, 1e+5, size=array_size)
                    s2 = np.random.uniform(-1e+5, 1e+5, size=array_size)

                    print('Encrypting arrays')
                    try:
                        s1_enc = tenseal_sp.encrypt(s1, log=True)
                        s2_enc = tenseal_sp.encrypt(s2, log=False)
                    except ValueError:
                        mlflow.log_param('failed', 1)
                        continue

                    print('Addition')
                    sum_enc_s1s2 = tenseal_sp.add_cypher_cypher(s1_enc, s2_enc)

                    print('Decrypting arrays')
                    sum_enc_s1s2_decr = tenseal_sp.decrypt(sum_enc_s1s2, log=False)
                    s1_enc_decr = tenseal_sp.decrypt(s1_enc, log=True)
                    s2_enc_decr = tenseal_sp.decrypt(s2_enc, log=False)

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
                        print('Multiplying')
                        try:
                            mult_enc_s1s2 = tenseal_sp.multiply_plain_cypher(plain_arr, s1_enc)
                            print('Decrypting arrays')
                            mult_enc_s1_decr = tenseal_sp.decrypt(mult_enc_s1s2, log=False)

                            mult_s1 = np.dot(plain_arr, s1)

                            all_cl1 = np.allclose(mult_s1, mult_enc_s1_decr, rtol=1e-05, atol=1e-08, equal_nan=False)
                            diff = max_abs_diff(mult_s1, mult_enc_s1_decr)
                        except ValueError:
                            all_cl1 = -1
                            diff = -1

                        mlflow.log_metric(
                            f'close_decr_mult_{plain_size}',
                            all_cl1
                        )

                        mlflow.log_metric(
                            f'max_abs_diff_mult_{plain_size}',
                            diff
                        )


if __name__ == '__main__':
    main()
