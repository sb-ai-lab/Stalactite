import logging
import os
import uuid
import random
import threading
from threading import Thread
from typing import List

import torch
import mlflow
import scipy as sp
from sklearn.metrics import mean_absolute_error
from datasets import DatasetDict

from stalactite.models.linreg_batch import LinearRegressionBatch
from stalactite.data_loader import load, init, DataPreprocessor
from stalactite.batching import ListBatcher
from stalactite.metrics import ComputeAccuracy_numpy
from stalactite.party_member_impl import PartyMemberImpl
from stalactite.party_master_impl import PartyMasterImpl, PartyMasterImplConsequently
from stalactite.communications.local import LocalMasterPartyCommunicator, LocalMemberPartyCommunicator
from stalactite.base import PartyMember


logging.basicConfig(level=logging.DEBUG)
logging.getLogger("urllib3").setLevel(logging.CRITICAL)
logger = logging.getLogger(__name__)


def run_single_member(exp_uid: str, datasets_list: List[DatasetDict], member_id: int, ds_size: int,
                      input_dims: List[int], batch_size: int, epochs: int,
                      members_count: int):
    with mlflow.start_run():

        log_params = {
            "ds_size": ds_size,
            "batch_size": batch_size,
            "epochs": epochs,
            "mode": "single",
            "member_id": member_id,
            "exp_uid": exp_uid,
            "members_count": members_count

        }
        mlflow.log_params(log_params)

        dataset = datasets_list[member_id]
        # extract the input data matrix and the targets
        X_train_all = dataset["train_train"][f"image_part_{member_id}"][:ds_size]
        Y_train_all = dataset["train_train"]["label"][:ds_size]

        X_test = dataset["train_val"][f"image_part_{member_id}"]
        Y_test = dataset["train_val"]["label"]

        model = LinearRegressionBatch(input_dim=input_dims[member_id], output_dim=1, reg_lambda=0.5)
        batcher = ListBatcher(epochs=epochs, members=[str(x) for x in range(10)], uids=[str(x) for x in range(ds_size)], batch_size=batch_size)

        step = 0
        for i, titer in enumerate(batcher):
            step += 1
            logger.debug(f"batch: {i}")
            batch = titer.batch
            tensor_idx = [int(x) for x in batch]
            X_train = dataset["train_train"][f"image_part_{member_id}"][tensor_idx]
            Y_train = dataset["train_train"]["label"][tensor_idx]

            U, S, Vt = sp.linalg.svd(X_train, full_matrices=False, overwrite_a=False, check_finite=False)
            model.update_weights(data_U=U, data_S=S, data_Vh=Vt, rhs=Y_train)

            train_predictions = model.predict(X_train_all)
            train_predictions = torch.mean(torch.stack((train_predictions,), dim=1), dim=1)

            test_predictions = model.predict(X_test)
            test_predictions = torch.mean(torch.stack((test_predictions,), dim=1), dim=1)

            train_mae = mean_absolute_error(Y_train_all.numpy(), train_predictions.numpy())
            test_mae = mean_absolute_error(Y_test.numpy(), test_predictions.numpy())

            acc = ComputeAccuracy_numpy()
            train_acc = acc.compute(true_label=Y_train_all.numpy(), predictions=train_predictions.numpy())
            test_acc = acc.compute(true_label=Y_test.numpy(), predictions=test_predictions.numpy())

            mlflow.log_metric("train_mae", train_mae, step=step)
            mlflow.log_metric("train_acc", train_acc, step=step)

            mlflow.log_metric("test_mae", test_mae, step=step)
            mlflow.log_metric("test_acc", test_acc, step=step)


def run_vfl(exp_uid: str, params, datasets_list: List[DatasetDict], members_count: int, ds_size: int,
            input_dims: List[int], batch_size: int, epochs: int,
            is_consequently: bool):
    with mlflow.start_run():

        log_params = {
            "ds_size": ds_size,
            "batch_size": batch_size,
            "epochs": epochs,
            "mode": "vfl",
            "members_count": members_count,
            "exp_uid": exp_uid,
            "is_consequently": is_consequently

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
                model=LinearRegressionBatch(input_dim=input_dims[i], output_dim=1, reg_lambda=0.2),
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
    mlflow.set_experiment(os.environ.get("EXPERIMENT", "local_vert_updated"))
    # models input dims for 1, 2, 3 and 5 members
    input_dims_list = [[619], [304, 315], [204, 250, 165], [], [108, 146, 150, 147, 68]]

    ds_size = int(os.environ.get("DS_SIZE", 1000))
    is_consequently = bool(int(os.environ.get("IS_CONSEQUENTLY")))

    batch_size = int(os.environ.get("BATCH_SIZE", 500))
    epochs = int(os.environ.get("EPOCHS", 1))
    mode = os.environ.get("MODE", "single")
    members_count = int(os.environ.get("MEMBERS_COUNT", 3))

    params = init()
    for m in range(members_count):
        params[m].data.dataset = f"mnist_binary38_parts{members_count}"
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
                ds_size=ds_size, epochs=epochs, input_dims=input_dims_list[members_count-1],
                members_count=members_count
            )

    elif mode.lower() == "vfl":
        run_vfl(
            exp_uid=exp_uid, params=params, datasets_list=datasets_list, members_count=members_count,
            batch_size=batch_size, ds_size=ds_size, epochs=epochs, input_dims=input_dims_list[members_count-1],
            is_consequently=is_consequently
            )
    else:
        raise ValueError(f"Unrecognized mode: {mode}. Please choose one of the following: single or vfl")


if __name__ == "__main__":
    main()
