import logging
import os
import uuid
import random
import threading
from threading import Thread
from typing import List

import datasets
import torch
import mlflow
import scipy as sp
from sklearn.metrics import mean_absolute_error
from datasets import DatasetDict

from stalactite.models.linreg_batch import LinearRegressionBatch
from stalactite.models.logreg_batch import LogisticRegressionBatch
from stalactite.data_loader import load, init, DataPreprocessor
from stalactite.data_preprocessors import FullDataTensor
from stalactite.batching import ListBatcher
from stalactite.metrics import ComputeAccuracy_numpy
from stalactite.party_member_impl import PartyMemberImpl
from stalactite.party_master_impl import PartyMasterImpl, PartyMasterImplConsequently, PartyMasterImplLogreg
from stalactite.communications.local import LocalMasterPartyCommunicator, LocalMemberPartyCommunicator
from stalactite.base import PartyMember


logging.basicConfig(level=logging.DEBUG)
logging.getLogger("urllib3").setLevel(logging.CRITICAL)
logger = logging.getLogger(__name__)


def run_single_member(exp_uid: str, datasets_list: List[DatasetDict], member_id: int, ds_size: int,
                      input_dims: List[int], batch_size: int, epochs: int,
                      members_count: int, model_name: str, lr: float, ds_name: str):
    with mlflow.start_run():

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
            is_multilabel = True
        elif ds_name == "multilabel":
            features_key = "features_part_"
            labels_key = "labels"
            output_dim = 10
            is_multilabel = True
        else:
            raise ValueError("Unknown dataset")

        X_train_all = dataset["train_train"][f"{features_key}{member_id}"][:ds_size]
        Y_train_all = dataset["train_train"][labels_key][:ds_size]

        X_test = dataset["train_val"][f"{features_key}{member_id}"]
        Y_test = dataset["train_val"][labels_key]

        if model_name == "linreg":
            model = LinearRegressionBatch(input_dim=input_dims[member_id], output_dim=output_dim, reg_lambda=0.5)
        elif model_name == "logreg":
            model = LogisticRegressionBatch(input_dim=input_dims[member_id], output_dim=output_dim, learning_rate=lr)
        else:
            raise ValueError(f"unknown model name: {model_name}, set one of ['linreg', 'logreg']")

        batcher = ListBatcher(
            epochs=epochs,
            members=[str(x) for x in range(10)], uids=[str(x) for x in range(ds_size)],
            batch_size=batch_size
        )

        step = 0
        for i, titer in enumerate(batcher):
            step += 1
            logger.debug(f"batch: {i}")
            batch = titer.batch
            tensor_idx = [int(x) for x in batch]
            X_train = dataset["train_train"][f"{features_key}{member_id}"][tensor_idx]
            Y_train = dataset["train_train"][labels_key][tensor_idx]

            if model_name == "linreg":

                U, S, Vt = sp.linalg.svd(X_train, full_matrices=False, overwrite_a=False, check_finite=False)
                model.update_weights(data_U=U, data_S=S, data_Vh=Vt, rhs=Y_train)
            else:
                model.update_weights(X_train, Y_train, is_single=True)

            if model_name == "logreg":
                train_predictions = torch.sigmoid(model.predict(X_train_all))
                test_predictions = torch.sigmoid(model.predict(X_test))

            else:
                train_predictions = model.predict(X_train_all)
                test_predictions = model.predict(X_test)

            train_mae = mean_absolute_error(Y_train_all.numpy(), train_predictions.detach().numpy())
            test_mae = mean_absolute_error(Y_test.numpy(), test_predictions.detach().numpy())

            acc = ComputeAccuracy_numpy(is_linreg=True if model_name == "linreg" else False, is_multilabel=is_multilabel)
            train_acc = acc.compute(true_label=Y_train_all.numpy(), predictions=train_predictions.detach().numpy())
            test_acc = acc.compute(true_label=Y_test.numpy(), predictions=test_predictions.detach().numpy())

            mlflow.log_metric("train_mae", train_mae, step=step)
            mlflow.log_metric("train_acc", train_acc, step=step)

            mlflow.log_metric("test_mae", test_mae, step=step)
            mlflow.log_metric("test_acc", test_acc, step=step)


def run_vfl(exp_uid: str, params, datasets_list: List[DatasetDict], members_count: int, ds_size: int,
            input_dims: List[int], batch_size: int, epochs: int,
            is_consequently: bool, model_name: str, lr: float, ds_name: str):
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
        mlflow.log_params(log_params)
        shared_uids_count = ds_size
        num_dataset_records = [200 + random.randint(100, 1000) for _ in range(members_count)]
        shared_record_uids = [str(i) for i in range(shared_uids_count)]
        target_uids = shared_record_uids
        targets = datasets_list[0]["train_train"]["label"][:ds_size]
        test_targets = datasets_list[0]["train_val"]["label"]
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
                run_mlflow=True
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
                run_mlflow=True
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
                run_mlflow=True
            )

        members = [
            PartyMemberImpl(
                uid=f"member-{i}",
                model_update_dim_size=input_dims[i],
                member_record_uids=member_uids,
                model=LinearRegressionBatch(input_dim=input_dims[i], output_dim=1, reg_lambda=0.2)
                if model_name == "linreg" else LogisticRegressionBatch(input_dim=input_dims[i], output_dim=1,
                                                                       learning_rate=lr),
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
    mlflow.set_experiment(os.environ.get("EXPERIMENT", "local_vert_logreg"))

    ds_name = os.environ.get("DATASET", "MNIST")
    ds_size = int(os.environ.get("DS_SIZE", 1000))
    is_consequently = bool(int(os.environ.get("IS_CONSEQUENTLY")))
    batch_size = int(os.environ.get("BATCH_SIZE", 500))
    epochs = int(os.environ.get("EPOCHS", 1))
    mode = os.environ.get("MODE", "single")
    members_count = int(os.environ.get("MEMBERS_COUNT", 3))
    model_name = os.environ.get("MODEL_NAME")
    learning_rate = float(os.environ.get("LR", 0.01))

    # models input dims for 1, 2, 3 and 5 members
    if ds_name == "MNIST":
        input_dims_list = [[619], [304, 315], [204, 250, 165], [], [108, 146, 150, 147, 68]]
    else:
        input_dims_list = [[100], [40, 60], [20, 50, 30], [], [10, 5, 45, 15, 25]]

    params = init(config_path="../experiments/configs/config_local_multilabel.yaml")

    for m in range(members_count):
        if model_name == "linreg":
            params[m].data.dataset = f"mnist_binary38_parts{members_count}"
        elif model_name == "logreg":
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
    else:
        raise ValueError(f"Unknown dataset: {ds_name}, choose one from ['mnist', 'multilabel']")


    # with open(f"datasets_{members_count}_members.pkl", "wb") as fp:
    #     pickle.dump({"data": datasets_list}, fp)
    exp_uid = str(uuid.uuid4())
    if mode.lower() == "single":
        for member_id in range(members_count):
            logger.info("starting experiment for member: " + str(member_id))
            run_single_member(
                exp_uid=exp_uid, datasets_list=datasets_list, member_id=member_id, batch_size=batch_size,
                ds_size=ds_size, epochs=epochs, input_dims=input_dims_list[members_count-1],
                members_count=members_count, model_name=model_name, lr=learning_rate, ds_name=ds_name
            )

    elif mode.lower() == "vfl":
        run_vfl(
            exp_uid=exp_uid, params=params, datasets_list=datasets_list, members_count=members_count,
            batch_size=batch_size, ds_size=ds_size, epochs=epochs, input_dims=input_dims_list[members_count-1],
            is_consequently=is_consequently, model_name=model_name, lr=learning_rate, ds_name=ds_name
            )
    else:
        raise ValueError(f"Unrecognized mode: {mode}. Please choose one of the following: single or vfl")


if __name__ == "__main__":
    main()
