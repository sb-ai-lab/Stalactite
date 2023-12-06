import logging
import math
import random
import threading
import uuid
from threading import Thread

import torch

from stalactite.base import PartyMember
from stalactite.communications.local import LocalMasterPartyCommunicator, LocalMemberPartyCommunicator
from stalactite.party_member_impl import PartyMemberImpl
from stalactite.party_master_impl import PartyMasterImpl
from data_loader import load, init, DataPreprocessor
from models.linreg_batch import LinearRegressionBatch

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)


def main():
    # ds_sample = 10_000
    # members_count = 3
    # params = init()
    # dataset, _ = load(params)
    # datasets_list = []
    # for m in range(members_count):
    #     logger.info(f"preparing dataset for member: {m}")
    #     params[m]["data"]["features_key"] = f'image_part_{m}'
    #     dp = DataPreprocessor(dataset, params[m].data, member_id=m)  # todo: move it inside member
    #     tmp_dataset, _ = dp.preprocess()  # todo: I should do this for updateting params (add input_dim and output_dim for model)
    #     datasets_list.append(tmp_dataset)
    # # logger.info(updated_data_params) #todo: check why func whith return values is so strange and return nothing
    #
    # y_list = datasets_list[0]["train_train"]["label"][:ds_sample]
    # # shapes = [d["train_train"][f"image_part_{i}"].size() for i, d in enumerate(datasets_list)]
    # input_dims = [204, 250, 165]
    # master = PartyMasterImpl(
    #     epochs=1,
    #     report_train_metrics_iteration=1,
    #     report_test_metrics_iteration=1,
    #     Y=y_list
    # )
    #
    # # input_dim = 204
    # output_dim = 1
    # members = [
    #     PartyMemberImpl(
    #         model=LinearRegressionBatch(input_dim=input_dims[m], output_dim=output_dim, reg_lambda=0.1),
    #         dataset=datasets_list[m], member_id=m) for m in range(members_count)
    # ]
    # party = PartyImpl(master, members, ds_sample=ds_sample)


    members_count = 3
    epochs = 1
    # shared uids also identifies dataset size
    # 1. target should be supplied with record uids
    # 2. each member should have their own data (of different sizes)
    # 3. uids mathching should be performed with relation to uids available for targets on master
    # 4. target is also subject for filtering on master
    # 5. shared_uids should identify uids that shared by all participants (including master itself)
    # and should be generated beforehand

    shared_uids_count = 10000  # это типа пользователи
    ds_sample = 10000
    batch_size = 1000

    # model_update_dim_size = 5
    # master + all members
    # num_target_records = 1000  # сколько таргетов

    num_dataset_records = [200 + random.randint(100, 1000) for _ in range(members_count)] #[205, 506, 777] # размеры датасетов у каждого мембера
    shared_record_uids = [str(i) for i in range(shared_uids_count)] # список типа пользователей
    # target_uids = [
    #     *shared_record_uids,
    #     *(str(uuid.uuid4()) for _ in range(num_target_records - len(shared_record_uids)))
    # ] # идентификаторы пользователей (которые нам и нужны и не нужны)

    target_uids = shared_record_uids  # для упрощения
    members_datasets_uids = [
        [*shared_record_uids, *(str(uuid.uuid4()) for _ in range(num_records - len(shared_record_uids)))]
        for num_records in num_dataset_records
    ] #идентификаторы пользователей на датасетах мемберов, включающие в себя нужные и не нужные

    shared_party_info = dict()

    #my part starts here
    params = init()
    dataset, _ = load(params)
    datasets_list = []
    for m in range(members_count):
        logger.info(f"preparing dataset for member: {m}")
        dp = DataPreprocessor(dataset, params[m].data, member_id=m)  # todo: move it inside member
        tmp_dataset, _ = dp.preprocess()
        datasets_list.append(tmp_dataset)

    targets = datasets_list[0]["train_train"]["label"][:ds_sample]
    input_dims = [204, 250, 165]

    master = PartyMasterImpl(
        uid="master",
        epochs=epochs,
        report_train_metrics_iteration=1,
        report_test_metrics_iteration=1,
        target=targets, #torch.rand(shared_uids_count),
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
