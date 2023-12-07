import logging
import os
import uuid
import random
import threading
from threading import Thread

import torch
import mlflow
import numpy as np
import scipy as sp
from sklearn.metrics import mean_absolute_error

from stalactite.models.linreg_batch import LinearRegressionBatch
from stalactite.data_loader import load, init, DataPreprocessor
from stalactite.batching import ListBatcher
from stalactite.metrics import ComputeAccuracy_numpy
from stalactite.party_member_impl import PartyMemberImpl
from stalactite.party_master_impl import PartyMasterImpl
from stalactite.communications.local import LocalMasterPartyCommunicator, LocalMemberPartyCommunicator
from stalactite.base import PartyMember

# from experiment_utils import (
#     get_datasets,
#     get_spark_configs_as_dict,
#     check_number_of_allocated_executors,
#     get_partition_num,
#     prepare_datasets, get_models,
# )

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_single_member(exp_uid: str, datasets_list, member_id: int, ds_size, input_dims, batch_size, epochs):
    with mlflow.start_run():

        log_params = {
            "ds_size": ds_size,
            "batch_size": batch_size,
            "epochs": epochs,
            "mode": "single",
            "member_id": member_id,
            "exp_uid": exp_uid,

        }
        mlflow.log_params(log_params)

        dataset = datasets_list[member_id]
        # extract the input data matrix and the targets
        X_train_all = dataset["train_train"][f"image_part_{member_id}"][:ds_size]
        Y_train_all = dataset["train_train"]["label"][:ds_size]

        X_val = dataset["train_val"][f"image_part_{member_id}"]
        Y_val = dataset["train_val"]["label"]

        model = LinearRegressionBatch(input_dim=input_dims[member_id], output_dim=1, reg_lambda=0.5)
        batcher = ListBatcher(batch_size=batch_size, uids=[str(x) for x in range(ds_size)])

        step = 0
        for epoch in range(epochs):
            for i, batch in enumerate(batcher):
                step += 1
                logger.debug(f"batch: {i}")
                tensor_idx = [int(x) for x in batch]
                X_train = dataset["train_train"][f"image_part_{member_id}"][tensor_idx]
                Y_train = dataset["train_train"]["label"][tensor_idx]

                U, S, Vt = sp.linalg.svd(X_train, full_matrices=False, overwrite_a=False, check_finite=False)
                model.update_weights(data_U=U, data_S=S, data_Vh=Vt, rhs=Y_train)

                train_predictions = model.predict(X_train_all)
                train_predictions = torch.mean(torch.stack((train_predictions,), dim=1), dim=1)

                val_predictions = model.predict(X_val)
                val_predictions = torch.mean(torch.stack((val_predictions,), dim=1), dim=1)

                train_mae = mean_absolute_error(Y_train_all.numpy(), train_predictions.numpy())
                val_mae = mean_absolute_error(Y_val.numpy(), val_predictions.numpy())

                acc = ComputeAccuracy_numpy()
                train_acc = acc.compute(true_label=Y_train_all.numpy(), predictions=train_predictions.numpy())
                val_acc = acc.compute(true_label=Y_val.numpy(), predictions=val_predictions.numpy())


                mlflow.log_metric("train_mse", train_mae, step=step)
                mlflow.log_metric("train_acc", train_acc, step=step)

                mlflow.log_metric("val_mse", val_mae, step=step)
                mlflow.log_metric("val_acc", val_acc, step=step)


def run_vfl(exp_uid: str, params, datasets_list, members_count: int, ds_size, input_dims, batch_size, epochs):
    with mlflow.start_run():

        log_params = {
            "ds_size": ds_size,
            "batch_size": batch_size,
            "epochs": epochs,
            "mode": "vfl",
            "members_count": members_count,
            "exp_uid": exp_uid,

        }
        mlflow.log_params(log_params)
        shared_uids_count = ds_size
        num_dataset_records = [200 + random.randint(100, 1000) for _ in range(members_count)]
        shared_record_uids = [str(i) for i in range(shared_uids_count)]
        target_uids = shared_record_uids
        targets = datasets_list[0]["train_train"]["label"][:ds_size]
        members_datasets_uids = [
            [*shared_record_uids, *(str(uuid.uuid4()) for _ in range(num_records - len(shared_record_uids)))]
            for num_records in num_dataset_records
        ]
        shared_party_info = dict()

        master = PartyMasterImpl(
            uid="master",
            epochs=epochs,
            report_train_metrics_iteration=1,
            report_test_metrics_iteration=1,
            target=targets,
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
                model=LinearRegressionBatch(input_dim=input_dims[i], output_dim=1, reg_lambda=0.5),
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


def main(dataset_name: str):

    k = int(os.environ.get("K", 100))
    mlflow_tracking_uri = os.environ.get(
        "MLFLOW_TRACKING_URI", "http://node16.bdcl:9876"
    )

    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(os.environ.get("EXPERIMENT", "local"))


    input_dims = [204, 250, 165]

    ds_size = int(os.environ.get("DS_SIZE", 1000))
    batch_size = int(os.environ.get("BATCH_SIZE", 500))
    epochs = int(os.environ.get("EPOCHS", 1))
    mode = os.environ.get("MODE", "single")
    members_count = int(os.environ.get("MEMBERS_COUNT", 3))

    params = init()
    dataset, _ = load(params)
    datasets_list = []
    for m in range(members_count):
        logger.info(f"preparing dataset for member: {m}")
        dp = DataPreprocessor(dataset, params[m].data, member_id=m)
        tmp_dataset, _ = dp.preprocess()
        datasets_list.append(tmp_dataset)

    exp_uid = str(uuid.uuid4())
    if mode.lower() == "single":

        for member_id in range(members_count):
            logger.info("starting experiment for member: " + str(member_id))
            run_single_member(
                exp_uid=exp_uid, datasets_list=datasets_list, member_id=member_id, batch_size=batch_size,
                ds_size=ds_size, epochs=epochs, input_dims=input_dims
            )

    elif mode.lower() == "vfl":
        run_vfl(
            exp_uid=exp_uid, params=params, datasets_list=datasets_list, members_count=members_count,
            batch_size=batch_size, ds_size=ds_size, epochs=epochs, input_dims=input_dims,
            )
    else:
        raise ValueError(f"Unrecognized mode: {mode}. Please choose one of the following: single or vfl")


if __name__ == "__main__":
    dataset = os.environ.get("DATASET", "MNIST")
    main(dataset_name=dataset)
