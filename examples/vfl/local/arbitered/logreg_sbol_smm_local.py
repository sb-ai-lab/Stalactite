# from examples.utils.local_arbitered_experiment import run
from examples.utils.arbitered_local_experiment_single import run
from examples.utils.helpers import path_from_root

CONFIG_PATH = path_from_root('examples/configs/arbitered-logreg-sbol-smm-local.yml')

if __name__ == "__main__":
    run(CONFIG_PATH)
