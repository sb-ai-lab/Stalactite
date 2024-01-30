import logging
import os
from pathlib import Path
import pickle

import mlflow
import click
import numpy as np
import torch

from stalactite.configs import VFLConfig
from stalactite.party_single_impl import PartySingleLinreg, PartySingleLogreg, PartySingleLogregSklearn

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
sh = logging.StreamHandler()
sh.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
logger.addHandler(sh)


def compute_class_weights(classes_idx, y_train) -> torch.Tensor:
    pos_weights_list = []
    for i, c_idx in enumerate(classes_idx):
        unique, counts = np.unique(y_train[:, i], return_counts=True)
        if unique.shape[0] < 2:
            raise ValueError(f"class {c_idx} has no label 1")
        pos_weights_list.append(counts[0] / counts[1])
    return torch.tensor(pos_weights_list)


def compute_class_distribution(classes_idx: list, y: torch.Tensor, name: str) -> None:
    logger.info(f"{name} distribution")
    for i, c_idx in enumerate(classes_idx):
        unique, counts = np.unique(y[:, i], return_counts=True)
        logger.info(f"for class: {c_idx}")
        logger.info(np.asarray((unique, counts)).T)
        if unique.shape[0] < 2:
            raise ValueError(f"class {c_idx} has no label 1")



@click.option(
    "--config-path", type=str, default=None, required=False, help="Path to the configuration file in `YAML` format."
)
def main(config_path: str = None):
    if config_path is None:
        config_path = os.environ.get(
            'SINGLE_MEMBER_CONFIG_PATH',
            os.path.join(Path(__file__).parent.parent.parent, 'configs/config-single.yml')
        )
    config = VFLConfig.load_and_validate(config_path)

    print(config)
    # Config move TODO
    ds_name = os.environ.get("DATASET", "MNIST")
    sample_size = int(os.environ.get("SAMPLE_SIZE", 10000))
    train_size = int(os.environ.get("TRAIN_SIZE", 1000))
    is_consequently = bool(int(os.environ.get("IS_CONSEQUENTLY", 0)))
    batch_size = int(os.environ.get("BATCH_SIZE", 500))
    epochs = int(os.environ.get("EPOCHS", 6))
    mode = os.environ.get("MODE", "single")
    members_count = int(os.environ.get("MEMBERS_COUNT", 3))
    model_name = os.environ.get("MODEL_NAME", 'linreg')
    learning_rate = float(os.environ.get("LR", 0.01))
    use_smm = bool(int(os.environ.get("USE_SMM", 1)))

    ds_size = config.data.dataset_size
    exp_uid = config.common.experiment_label

    classes_idx = [x for x in range(19)]
    # remove classes with low positive targets rate
    classes_idx = [x for x in classes_idx if x not in [18, 3, 12, 14]]

    n_labels = len(classes_idx)  # TODO remove

    member_id = 0 # TODO config

    if config.master.run_mlflow:
        mlflow.set_tracking_uri(f"http://{config.prerequisites.mlflow_host}:{config.prerequisites.mlflow_port}")
        mlflow.set_experiment(config.common.experiment_label)
        mlflow.start_run()

        log_params = {
            "ds_size": ds_size,
            "batch_size": batch_size,
            "epochs": epochs,
            "mode": "single",
            "member_id": member_id,
            "exp_uid": exp_uid,
            "members_count": members_count,
            "model_name": model_name,
            "learning_rate": learning_rate,
            "dataset": ds_name,
            "n_labels": n_labels
        }
        mlflow.log_params(log_params)

    with open(
        os.path.join(
            os.path.expanduser(config.data.host_path_data_dir), f"datasets_mnist_single.pkl"
        ),
        "rb",
    ) as f:
        datasets_list = pickle.load(f)["data"]

    # dataset = datasets_list[member_id]
    features_key = "image_part_"
    labels_key = "labels"
    output_dim = 1
    is_multilabel = False


    # unique, counts = np.unique(y_train_all, return_counts=True)
    # logger.info("train classes distribution")
    # logger.info(np.asarray((unique, counts)).T)

    class_weights = None # compute_class_weights(classes_idx, y_train_all) if use_class_weights else
    if config.master.run_mlflow:
        mlflow.log_param("class_weights", class_weights)






#
    # x_test = dataset["train_val"][f"{features_key}{member_id}"]
    # y_test = dataset["train_val"][labels_key][:, classes_idx]
#
    # compute_class_distribution(classes_idx=classes_idx, y=y_test, name="test")
    compute_inner_users = False # TODO?

    # if compute_inner_users:
    #     with open(f"test_inner_users_sbol_smm_sample{sample_size}.pkl", "rb") as f:
    #         test_inner_users = pickle.load(f)
    #     mlflow.log_param("test_inner_users", len(test_inner_users))
    #     compute_class_distribution(classes_idx=classes_idx, y=y_test[test_inner_users], name="test_inner_users")

    if model_name == "linreg":
        party = PartySingleLinreg(
            dataset=datasets_list,
            dataset_size=config.data.dataset_size,
            epochs=epochs,
            batch_size=batch_size,
            features_key=features_key,
            labels_key=labels_key,
            output_dim=output_dim,
            use_mlflow=config.master.run_mlflow,
            test_inner_users=None,
            report_train_metrics_iteration=config.common.report_train_metrics_iteration,
            report_test_metrics_iteration=config.common.report_test_metrics_iteration,
            is_multilabel=is_multilabel,
            input_dims=[619],
            learning_rate=learning_rate,
        )

        party.run()
    if config.master.run_mlflow:
        mlflow.end_run()


if __name__ == '__main__':
    main()

