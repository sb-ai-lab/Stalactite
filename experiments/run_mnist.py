import logging
import math
import random
import threading
import uuid
from threading import Thread

from stalactite.base import PartyMember
from stalactite.communications.local import LocalMasterPartyCommunicator, LocalMemberPartyCommunicator
from stalactite.party_member_impl import PartyMemberImpl
from stalactite.party_master_impl import PartyMasterImpl
from stalactite.data_loader import load, init, DataPreprocessor
from stalactite.models.linreg_batch import LinearRegressionBatch

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)


def main():

    members_count = 3
    epochs = 3

    shared_uids_count = 1000
    ds_sample = 1000
    batch_size = 100

    num_dataset_records = [200 + random.randint(100, 1000) for _ in range(members_count)]
    shared_record_uids = [str(i) for i in range(shared_uids_count)]
    target_uids = shared_record_uids
    members_datasets_uids = [
        [*shared_record_uids, *(str(uuid.uuid4()) for _ in range(num_records - len(shared_record_uids)))]
        for num_records in num_dataset_records
    ]

    shared_party_info = dict()

    params = init()
    dataset, _ = load(params)
    datasets_list = []
    for m in range(members_count):
        logger.info(f"preparing dataset for member: {m}")
        dp = DataPreprocessor(dataset, params[m].data, member_id=m)
        tmp_dataset, _ = dp.preprocess()
        datasets_list.append(tmp_dataset)

    targets = datasets_list[0]["train_train"]["label"][:ds_sample]
    input_dims = [204, 250, 165]

    master = PartyMasterImpl(
        uid="master",
        epochs=epochs,
        report_train_metrics_iteration=1,
        report_test_metrics_iteration=1,
        target=targets,
        target_uids=target_uids,
        batch_size=batch_size,
        model_update_dim_size=0
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

    assert master.iteration_counter == epochs * math.ceil(shared_uids_count / batch_size)
    assert all([member.iterations_counter == epochs * math.ceil(shared_uids_count / batch_size) for member in members])
    assert master.is_initialized and master.is_finalized
    assert all([member.is_initialized and member.is_finalized for member in members])


if __name__ == "__main__":
    main()
