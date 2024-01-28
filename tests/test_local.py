import logging
import math
import random
import threading
import uuid
from threading import Thread

import torch

from stalactite.base import PartyMember
from stalactite.communications.local import LocalMasterPartyCommunicator, LocalMemberPartyCommunicator
from stalactite.mocks import MockPartyMasterImpl, MockPartyMemberImpl


logging.basicConfig(level=logging.DEBUG)

logger = logging.getLogger(__name__)


def test_local_run():
    members_count = 3
    epochs = 5
    # shared uids also identifies dataset size
    # 1. target should be supplied with record uids
    # 2. each member should have their own data (of different sizes)
    # 3. uids mathching should be performed with relation to uids available for targets on master
    # 4. target is also subject for filtering on master
    # 5. shared_uids should identify uids that shared by all participants (including master itself)
    # and should be generated beforehand
    shared_uids_count = 100
    batch_size = 10
    model_update_dim_size = 5
    # master + all members
    num_target_records = 1000

    num_dataset_records = [200 + random.randint(100, 1000) for _ in range(members_count)]
    shared_record_uids = [str(i) for i in range(shared_uids_count)]
    target_uids = [
        *shared_record_uids,
        *(str(uuid.uuid4()) for _ in range(num_target_records - len(shared_record_uids)))
    ]
    members_datasets_uids = [
        [*shared_record_uids, *(str(uuid.uuid4()) for _ in range(num_records - len(shared_record_uids)))]
        for num_records in num_dataset_records
    ]

    shared_party_info = dict()

    master = MockPartyMasterImpl(
        uid="master",
        epochs=epochs,
        report_train_metrics_iteration=5,
        report_test_metrics_iteration=5,
        target=torch.rand(shared_uids_count),
        target_uids=target_uids,
        batch_size=batch_size,
        model_update_dim_size=model_update_dim_size
    )

    members = [
        MockPartyMemberImpl(
            uid=f"member-{i}",
            model_update_dim_size=model_update_dim_size,
            member_record_uids=member_uids
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
            shared_party_info=shared_party_info,
            master_id=master.id
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
