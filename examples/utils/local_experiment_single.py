import os
import logging
from pathlib import Path
from typing import Optional

import datasets

from stalactite.data_preprocessors import ImagePreprocessor, TabularPreprocessor
from stalactite.configs import VFLConfig
from stalactite.party_single_impl import PartySingleLinreg, PartySingleLogreg
from examples.utils.prepare_mnist import load_data as load_mnist
from examples.utils.prepare_sbol_smm import load_data as load_sbol_smm
from stalactite.helpers import reporting

logging.basicConfig(level=logging.DEBUG)
logging.getLogger("urllib3").setLevel(logging.CRITICAL)
logger = logging.getLogger(__name__)


def load_processors(config_path: str):
    config = VFLConfig.load_and_validate(config_path)

    if config.data.dataset.lower() == "mnist":

        if not os.path.exists(config.data.host_path_data_dir):
            load_mnist(config.data.host_path_data_dir, parts_num=1)

        dataset = {0: datasets.load_from_disk(
            os.path.join(f"{config.data.host_path_data_dir}/part_{0}")
        )}

        processors = [
            ImagePreprocessor(dataset=dataset[i], member_id=i, params=config) for i, v in dataset.items()
        ]

    elif config.data.dataset.lower() == "sbol_smm":

        if len(os.listdir(config.data.host_path_data_dir)) == 0:
            load_sbol_smm(os.path.dirname(config.data.host_path_data_dir), parts_num=1)

        dataset = {0: datasets.load_from_disk(
            os.path.join(f"{config.data.host_path_data_dir}/part_{0}")
        )}

        processors = [
            TabularPreprocessor(dataset=dataset[i], member_id=i, params=config) for i, v in dataset.items()
        ]

    else:
        raise ValueError(f"Unknown dataset: {config.data.dataset}, choose one from ['mnist', 'multilabel']")

    return processors


def run(config_path: Optional[str] = None):
    if config_path is None:
        config_path = os.environ.get(
            'SINGLE_MEMBER_CONFIG_PATH',
            os.path.join(Path(__file__).parent.parent.parent, 'configs/linreg-mnist-local.yml')
        )
    config = VFLConfig.load_and_validate(config_path)
    model_name = config.vfl_model.vfl_model_name

    with reporting(config):

        processors = load_processors(config_path)

        if model_name == "linreg":
            party = PartySingleLinreg(
                processor=processors[0],
                batch_size=config.vfl_model.batch_size,
                epochs=config.vfl_model.epochs,
                report_train_metrics_iteration=config.common.report_train_metrics_iteration,
                report_test_metrics_iteration=config.common.report_test_metrics_iteration,
                use_mlflow=config.master.run_mlflow
            )

        elif model_name == "logreg":
            party = PartySingleLogreg(
                processor=processors[0],
                batch_size=config.vfl_model.batch_size,
                epochs=config.vfl_model.epochs,
                report_train_metrics_iteration=config.common.report_train_metrics_iteration,
                report_test_metrics_iteration=config.common.report_test_metrics_iteration,
                use_mlflow=config.master.run_mlflow
            )
        else:
            raise ValueError(f"unknown model name: {model_name}")

        party.run()


if __name__ == "__main__":
    run()
