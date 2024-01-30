import os
from pathlib import Path

from examples.utils.local_experiment import run

CONFIG_PATH = os.path.join(Path(__file__).parent.parent.parent, 'configs/logreg-sbol-smm-local.yml')

if __name__ == "__main__":
    run(CONFIG_PATH)