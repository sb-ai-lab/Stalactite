import os
from pathlib import Path

from examples.utils.local_experiment import run

CONFIG_PATH = os.path.join(Path(__file__).parent.parent.parent, 'configs/linreg-mnist-seq-local.yml')

if __name__ == "__main__":
    run(CONFIG_PATH)
