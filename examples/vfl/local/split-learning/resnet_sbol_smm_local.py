from examples.utils.local_experiment import run
from examples.utils.helpers import path_from_root

CONFIG_PATH = path_from_root('examples/configs/resnet-splitNN-sbol-smm-local.yml')

if __name__ == "__main__":
    run(CONFIG_PATH)
