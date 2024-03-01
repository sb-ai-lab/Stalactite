from examples.utils.local_experiment_single import run
from examples.utils.helpers import path_from_root

CONFIG_PATH = path_from_root('examples/configs/logreg-sbol-smm-single.yml')

if __name__ == "__main__":
    run(CONFIG_PATH)
