import os
import logging
import pickle
import uuid
import random
import threading
from threading import Thread
from typing import List

import torch
import mlflow
import numpy as np
import datasets
import scipy as sp
from sklearn.metrics import mean_absolute_error,  roc_auc_score, precision_recall_curve, auc
from datasets import DatasetDict
from sklearn.linear_model import LogisticRegression as LogRegSklearn

from stalactite.models.linreg_batch import LinearRegressionBatch
from stalactite.models.logreg_batch import LogisticRegressionBatch
from stalactite.party_member_impl import PartyMemberImpl
from stalactite.data_loader import load, init, DataPreprocessor
from stalactite.batching import ListBatcher
from stalactite.metrics import ComputeAccuracy_numpy
from stalactite.party_master_impl import PartyMasterImpl, PartyMasterImplConsequently, PartyMasterImplLogreg
from stalactite.communications.local import LocalMasterPartyCommunicator, LocalMemberPartyCommunicator
from stalactite.base import PartyMember


logging.basicConfig(level=logging.DEBUG)
logging.getLogger("urllib3").setLevel(logging.CRITICAL)
logger = logging.getLogger(__name__)


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


def run_single_member(exp_uid: str, datasets_list: List[DatasetDict], member_id: int, ds_size: int,
                      input_dims: List[int], batch_size: int, epochs: int,
                      members_count: int, model_name: str, lr: float, ds_name: str, use_class_weights: bool = False,
                      compute_inner_users: bool = False, sample_size: int = 10_000):
    with mlflow.start_run():

        classes_idx = [x for x in range(19)]
        # remove classes with low positive targets rate
        classes_idx = [x for x in classes_idx if x not in [18, 3, 12, 14]]

        n_labels = len(classes_idx)
        mlflow.log_param("n_labels", n_labels)

        log_params = {
            "ds_size": ds_size,
            "batch_size": batch_size,
            "epochs": epochs,
            "mode": "single",
            "member_id": member_id,
            "exp_uid": exp_uid,
            "members_count": members_count,
            "model_name": model_name,
            "learning_rate": lr,
            "dataset": ds_name
        }
        mlflow.log_params(log_params)

        dataset = datasets_list[member_id]

        if ds_name == "mnist":
            features_key = "image_part_"
            labels_key = "label"
            output_dim = 1
            is_multilabel = False
        elif ds_name == "multilabel":
            features_key = "features_part_"
            labels_key = "labels"
            output_dim = 10
            is_multilabel = True
        elif ds_name in ["sbol", "sbol_smm"]:
            features_key = "features_part_"
            labels_key = "labels"
            output_dim = 19
            is_multilabel = True
        else:
            raise ValueError("Unknown dataset")

        x_train_all = dataset["train_train"][f"{features_key}{member_id}"][:ds_size]
        y_train_all = dataset["train_train"][labels_key][:ds_size][:, classes_idx]

        unique, counts = np.unique(y_train_all, return_counts=True)
        logger.info("train classes distribution")
        logger.info(np.asarray((unique, counts)).T)

        class_weights = compute_class_weights(classes_idx, y_train_all) if use_class_weights else None
        mlflow.log_param("class_weights", class_weights)

        x_test = dataset["train_val"][f"{features_key}{member_id}"]
        y_test = dataset["train_val"][labels_key][:, classes_idx]

        compute_class_distribution(classes_idx=classes_idx, y=y_test, name="test")

        if compute_inner_users:
            with open(f"test_inner_users_sbol_smm_sample{sample_size}.pkl", "rb") as f:
                test_inner_users = pickle.load(f)
            mlflow.log_param("test_inner_users", len(test_inner_users))
            compute_class_distribution(classes_idx=classes_idx, y=y_test[test_inner_users], name="test_inner_users")

        if model_name == "linreg":
            model = LinearRegressionBatch(input_dim=input_dims[member_id], output_dim=output_dim, reg_lambda=0.5)
        elif model_name == "logreg":
            model = LogisticRegressionBatch(input_dim=input_dims[member_id], output_dim=n_labels, learning_rate=lr,
                                            class_weights=class_weights, init_weights=0.005)
        elif model_name == "logreg_sklearn":
            model = LogRegSklearn(random_state=22, penalty=None, max_iter=2000, class_weight="balanced")
            mlflow.log_param("class_weights", "balanced")
        else:
            raise ValueError(f"unknown model name: {model_name}, set one of ['linreg', 'logreg']")

        batcher = ListBatcher(
            epochs=epochs,
            members=[str(x) for x in range(10)], uids=[str(x) for x in range(y_train_all.shape[0])],
            batch_size=batch_size,
            shuffle=False
        )

        step = 0
        for i, titer in enumerate(batcher):
            step += 1
            logger.debug(f"batch: {i}")
            batch = titer.batch
            tensor_idx = [int(x) for x in batch]

            x_train = x_train_all[tensor_idx]
            y_train = y_train_all[tensor_idx]

            if (titer.previous_batch is None) and (i > 0):
                break

            if model_name == "linreg":
                model.update_weights(X_train=x_train, rhs=y_train)
            elif model_name == "logreg":
                model.update_weights(x_train, y_train, is_single=True)
                train_predictions = torch.sigmoid(model.predict(x_train_all)).detach().numpy()
                test_predictions = torch.sigmoid(model.predict(x_test)).detach().numpy()

            elif model_name == "logreg_sklearn":
                model.fit(x_train.numpy(), y_train.numpy())
                train_predictions = model.predict(x_train_all)
                test_predictions = model.predict(x_test)

            else:
                ValueError("unknown model")

            train_mae = mean_absolute_error(y_train_all.numpy(), train_predictions)
            test_mae = mean_absolute_error(y_test.numpy(), test_predictions)

            acc = ComputeAccuracy_numpy(is_linreg=True if model_name == "linreg" else False, is_multilabel=is_multilabel)
            train_acc = acc.compute(true_label=y_train_all.numpy(), predictions=train_predictions)
            test_acc = acc.compute(true_label=y_test.numpy(), predictions=test_predictions)

            mlflow.log_metric("train_mae", train_mae, step=step)
            mlflow.log_metric("train_acc", train_acc, step=step)

            mlflow.log_metric("test_mae", test_mae, step=step)
            mlflow.log_metric("test_acc", test_acc, step=step)

            if n_labels > 1:

                for avg in ["macro", "micro"]:
                    mlflow.log_metric(f"train_roc_auc_{avg}",
                                      roc_auc_score(y_train_all.numpy(), train_predictions, average=avg), step=step,
                                      )
                    mlflow.log_metric(f"test_roc_auc_{avg}",
                                      roc_auc_score(y_test.numpy(), test_predictions, average=avg), step=step
                                      )

                    if compute_inner_users:
                        mlflow.log_metric(f"test_roc_auc_{avg}_inner",
                                          roc_auc_score(y_test.numpy()[test_inner_users, :],
                                                        test_predictions[test_inner_users, :],
                                                        average=avg), step=step
                                          )

                roc_auc_train_list, roc_auc_test_list = [], []
                if compute_inner_users:
                    roc_auc_test_list_inner = []
                for i in range(n_labels):
                    roc_auc_train_list.append(roc_auc_score(y_train_all.numpy()[:, i], train_predictions[:, i]))
                    roc_auc_test_list.append(roc_auc_score(y_test.numpy()[:, i], test_predictions[:, i]))
                    if compute_inner_users:
                        roc_auc_test_list_inner.append(
                            roc_auc_score(y_test.numpy()[test_inner_users, i], test_predictions[test_inner_users, i])
                        )

                for i, cls in enumerate(classes_idx):
                    mlflow.log_metric(f"train_roc_auc_cls_{cls}", roc_auc_train_list[i])
                    mlflow.log_metric(f"test_roc_auc_cls_{cls}", roc_auc_test_list[i])
                    if compute_inner_users:
                        mlflow.log_metric(f"test_roc_auc_cls_{cls}_inner", roc_auc_test_list_inner[i])

            else:

                mlflow.log_metric("train_roc_auc", roc_auc_score(y_train_all.numpy(), train_predictions), step=step)
                mlflow.log_metric("test_roc_auc", roc_auc_score(y_test.numpy(), test_predictions), step=step)

                p_train, r_train, _ = precision_recall_curve(y_train_all.numpy(), train_predictions)
                pr_auc_train = auc(r_train, p_train)
                mlflow.log_metric("train_pr_auc", pr_auc_train, step=step)

                p_test, r_test, _ = precision_recall_curve(y_test.numpy(), test_predictions)
                pr_auc_test = auc(r_test, p_test)
                mlflow.log_metric("test_pr_auc", pr_auc_test, step=step)


def run_vfl(exp_uid: str, params, datasets_list: List[DatasetDict], members_count: int, ds_size: int,
            input_dims: List[int], batch_size: int, epochs: int,
            is_consequently: bool, model_name: str, lr: float, ds_name: str, use_class_weights: bool = False):
    with mlflow.start_run():

        log_params = {
            "ds_size": ds_size,
            "batch_size": batch_size,
            "epochs": epochs,
            "mode": "vfl",
            "members_count": members_count,
            "exp_uid": exp_uid,
            "is_consequently": is_consequently,
            "model_name": model_name,
            "learning_rate": lr,
            "dataset": ds_name

        }

        label_col = "labels"

        classes_idx = [x for x in range(19)]
        # remove classes with low positive targets rate
        classes_idx = [x for x in classes_idx if x not in [18, 3, 12, 14]]

        n_labels = len(classes_idx)
        mlflow.log_param("n_labels", n_labels)

        mlflow.log_params(log_params)
        shared_uids_count = ds_size
        num_dataset_records = [200 + random.randint(100, 1000) for _ in range(members_count)]
        shared_record_uids = [str(i) for i in range(shared_uids_count)]
        target_uids = shared_record_uids
        targets = datasets_list[0]["train_train"][label_col][:ds_size][:, classes_idx]
        test_targets = datasets_list[0]["train_val"][label_col][:, classes_idx]

        class_weights = compute_class_weights(classes_idx, targets) if use_class_weights else None
        mlflow.log_param("class_weights", class_weights)

        compute_class_distribution(classes_idx=classes_idx, y=test_targets, name="test")

        members_datasets_uids = [
            [*shared_record_uids, *(str(uuid.uuid4()) for _ in range(num_records - len(shared_record_uids)))]
            for num_records in num_dataset_records
        ]
        shared_party_info = dict()
        if is_consequently:
            master = PartyMasterImplConsequently(
                uid="master",
                epochs=epochs,
                report_train_metrics_iteration=1,
                report_test_metrics_iteration=1,
                target=targets,
                test_target=test_targets,
                target_uids=target_uids,
                batch_size=batch_size,
                model_update_dim_size=0,
                run_mlflow=True,
                class_weights=class_weights
            )

        elif model_name == "logreg":
            master = PartyMasterImplLogreg(
                uid="master",
                epochs=epochs,
                report_train_metrics_iteration=1,
                report_test_metrics_iteration=1,
                target=targets,
                test_target=test_targets,
                target_uids=target_uids,
                batch_size=batch_size,
                model_update_dim_size=0,
                run_mlflow=True,
                class_weights=class_weights
            )
        else:
            master = PartyMasterImpl(
                uid="master",
                epochs=epochs,
                report_train_metrics_iteration=1,
                report_test_metrics_iteration=1,
                target=targets,
                test_target=test_targets,
                target_uids=target_uids,
                batch_size=batch_size,
                model_update_dim_size=0,
                run_mlflow=True,
                class_weights=class_weights
            )

        members = [
            PartyMemberImpl(
                uid=f"member-{i}",
                model_update_dim_size=input_dims[i],
                member_record_uids=member_uids,
                model=LinearRegressionBatch(input_dim=input_dims[i], output_dim=1, reg_lambda=0.2)
                if model_name == "linreg" else LogisticRegressionBatch(input_dim=input_dims[i], output_dim=n_labels,
                                                                       learning_rate=lr, class_weights=class_weights,
                                                                       init_weights=0.005),
                dataset=datasets_list[i],
                data_params=params[i]["data"]
            )
            for i, member_uids in enumerate(members_datasets_uids)
        ]

        def local_master_main():
            logger.info("Starting thread %s" % threading.current_thread().name)
            comm = LocalMasterPartyCommunicator(
                participant=master,
                world_size=members_count,
                shared_party_info=shared_party_info
            )
            comm.run()
            logger.info("Finishing thread %s" % threading.current_thread().name)

        def local_member_main(member: PartyMember):
            logger.info("Starting thread %s" % threading.current_thread().name)
            comm = LocalMemberPartyCommunicator(
                participant=member,
                world_size=members_count,
                shared_party_info=shared_party_info
            )
            comm.run()
            logger.info("Finishing thread %s" % threading.current_thread().name)

        threads = [
            Thread(name=f"main_{master.id}", daemon=True, target=local_master_main),
            *(
                Thread(
                    name=f"main_{member.id}",
                    daemon=True,
                    target=local_member_main,
                    args=(member,)
                )
                for member in members
            )
        ]

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()


def main():

    mlflow_tracking_uri = os.environ.get(
        "MLFLOW_TRACKING_URI", "http://node16.bdcl:9876"
    )

    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(os.environ.get("EXPERIMENT", "local_sbol"))

    ds_name = os.environ.get("DATASET", "MNIST")
    sample_size = int(os.environ.get("SAMPLE_SIZE", 10000))
    train_size = int(os.environ.get("TRAIN_SIZE", 1000))
    is_consequently = bool(int(os.environ.get("IS_CONSEQUENTLY")))
    batch_size = int(os.environ.get("BATCH_SIZE", 500))
    epochs = int(os.environ.get("EPOCHS", 1))
    mode = os.environ.get("MODE", "single")
    members_count = int(os.environ.get("MEMBERS_COUNT", 3))
    model_name = os.environ.get("MODEL_NAME")
    learning_rate = float(os.environ.get("LR", 0.01))
    use_smm = bool(int(os.environ.get("USE_SMM", 1)))
    use_class_weights = bool(int(os.environ.get("USE_CLASS_WEIGHTS", 1)))

    # models input dims for 1, 2, 3 and 5 members
    if ds_name.lower() == "mnist":
        input_dims_list = [[619], [304, 315], [204, 250, 165], [], [108, 146, 150, 147, 68]]
        params = init(config_path="../experiments/configs/config_local_mnist.yaml")
    elif ds_name.lower() == "sbol":
        smm = "smm_" if use_smm else ""
        dim = 1356 if smm == "smm_" else 1345
        input_dims_list = [[0], [1345, 11]]
        if mode == "single":
            input_dims_list = [[0], [dim, 0]]
        params = init(config_path="../experiments/configs/config_local_multilabel.yaml")
    else:
        input_dims_list = [[100], [40, 60], [20, 50, 30], [], [10, 5, 45, 15, 25]]
        params = init(config_path="../experiments/configs/config_local_multilabel.yaml")

    for m in range(members_count):
        if model_name == "linreg":
            params[m].data.dataset = f"mnist_binary38_parts{members_count}"
        elif model_name == "logreg" or "catboost":
            params[m].data.dataset = f"mnist_binary01_38_parts{members_count}"
        else:
            raise ValueError("Unknown model name {}".format(model_name))

    if ds_name.lower() == "mnist":
        dataset, _ = load(params)
        datasets_list = []
        for m in range(members_count):
            logger.info(f"preparing dataset for member: {m}")
            dp = DataPreprocessor(dataset, params[m].data, member_id=m)
            tmp_dataset, _ = dp.preprocess()
            datasets_list.append(tmp_dataset)

    elif ds_name.lower() == "multilabel":
        dataset = {}
        for m in range(members_count):
            dataset[m] = datasets.load_from_disk(f"/home/dmitriy/data/multilabel_ds_parts{members_count}/part_{m}")

        datasets_list = []
        for m in range(members_count):
            logger.info(f"preparing dataset for member: {m}")
            dp = DataPreprocessor(dataset, params[m].data, member_id=m)
            tmp_dataset, _ = dp.preprocess()
            datasets_list.append(tmp_dataset)

    elif ds_name.lower() == "sbol":
        if smm != "":
            ds_name = "sbol_smm"
        dataset = {}
        if mode == "vfl":
            ds_path = f"vfl_multilabel_sber_sample{sample_size}_parts{members_count}"
        else:
            ds_path = f"multilabel_sber_{smm}sample{sample_size}_parts{members_count}"
        for m in range(members_count):
            if mode == "single":
                m = 0
            dataset[m] = datasets.load_from_disk(
                f"/home/dmitriy/data/{ds_path}/part_{m}"
            )
        datasets_list = []
        for m in range(members_count):
            logger.info(f"preparing dataset for member: {m}")
            dp = DataPreprocessor(dataset, params[m].data, member_id=m)
            tmp_dataset, _ = dp.preprocess()
            datasets_list.append(tmp_dataset)
            if mode == "single":
                break
    else:
        raise ValueError(f"Unknown dataset: {ds_name}, choose one from ['mnist', 'multilabel']")

    exp_uid = str(uuid.uuid4())
    if mode.lower() == "single":
        for member_id in range(members_count):
            logger.info("starting experiment for member: " + str(member_id))
            run_single_member(
                exp_uid=exp_uid, datasets_list=datasets_list, member_id=member_id, batch_size=batch_size,
                ds_size=train_size, epochs=epochs, input_dims=input_dims_list[members_count-1],
                members_count=members_count, model_name=model_name, lr=learning_rate, ds_name=ds_name,
                use_class_weights=use_class_weights
            )
            break

    elif mode.lower() == "vfl":
        run_vfl(
            exp_uid=exp_uid, params=params, datasets_list=datasets_list, members_count=members_count,
            batch_size=batch_size, ds_size=train_size, epochs=epochs, input_dims=input_dims_list[members_count-1],
            is_consequently=is_consequently, model_name=model_name, lr=learning_rate, ds_name=ds_name,
            use_class_weights=use_class_weights
            )
    else:
        raise ValueError(f"Unrecognized mode: {mode}. Please choose one of the following: single or vfl")


if __name__ == "__main__":
    main()
