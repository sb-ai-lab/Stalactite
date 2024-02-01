import os
from pathlib import Path

from examples.utils.local_experiment_single import run

CONFIG_PATH = os.path.join(Path(__file__).parent.parent, 'configs/logreg-sbol-smm-single.yml')

if __name__ == "__main__":
    run(CONFIG_PATH)
