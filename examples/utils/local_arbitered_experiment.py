import os
import logging
from pathlib import Path
import threading
from typing import Optional
import datasets

from stalactite.ml import (
    ArbiteredPartyMasterLogReg,
    ArbiteredPartyMemberLogReg,
    PartyArbiterLogReg
)
from stalactite.ml.arbitered.security_protocols.paillier_sp import SecurityProtocolPaillier, SecurityProtocolArbiterPaillier
from stalactite.data_preprocessors import ImagePreprocessor, TabularPreprocessor
# from stalactite.party_master_impl import PartyMasterImpl, PartyMasterImplConsequently, PartyMasterImplLogreg
from stalactite.communications.local import ArbiteredLocalPartyCommunicator
from stalactite.base import PartyMember
from stalactite.configs import VFLConfig
from examples.utils.prepare_mnist import load_data as load_mnist
from examples.utils.prepare_sbol_smm import load_data as load_sbol_smm
from stalactite.helpers import reporting, run_local_agents

logging.basicConfig(level=logging.DEBUG)
logging.getLogger("urllib3").setLevel(logging.CRITICAL)
logging.getLogger('PIL').setLevel(logging.ERROR)
logger = logging.getLogger(__name__)


def load_processors(config: VFLConfig):
    """

    Assigns parameters to preprocessor class, which is selected depending on the type of dataset: MNIST or SBOL.
    If there is no data to run the experiment, downloads data after preprocessing.

    """
    if config.data.dataset.lower() == "mnist":

        if len(os.listdir(config.data.host_path_data_dir)) == 0:
            load_mnist(config.data.host_path_data_dir, config.common.world_size)

        dataset = {}
        for m in range(config.common.world_size):
            dataset[m] = datasets.load_from_disk(
                os.path.join(f"{config.data.host_path_data_dir}/part_{m}")
            )

        processors = [
            ImagePreprocessor(dataset=dataset[i], member_id=i, params=config) for i, v in dataset.items()
        ]

    elif config.data.dataset.lower() == "sbol_smm":

        dataset = {}
        if len(os.listdir(config.data.host_path_data_dir)) == 0:
            load_sbol_smm(os.path.dirname(config.data.host_path_data_dir), parts_num=config.common.world_size + 1)

        for m in range(config.common.world_size + 1):
            dataset[m] = datasets.load_from_disk(
                os.path.join(f"{config.data.host_path_data_dir}/part_{m}")
            )
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
            os.path.join(Path(__file__).parent.parent, 'configs/linreg-mnist-local.yml')
        )

    config = VFLConfig.load_and_validate(config_path)
    processors = load_processors(config)

    with reporting(config):
        target_uids = [str(i) for i in range(config.data.dataset_size)]

        shared_party_info = dict()
        master_class = ArbiteredPartyMasterLogReg
        member_class = ArbiteredPartyMemberLogReg
        arbiter_class = PartyArbiterLogReg

        arbiter = arbiter_class(
            uid="arbiter",
            epochs=config.vfl_model.epochs,
            batch_size=config.vfl_model.batch_size,
            security_protocol=SecurityProtocolArbiterPaillier(),
            learning_rate=config.vfl_model.learning_rate,
            momentum=0.0,
        )

        master = master_class(
            uid="master",
            epochs=config.vfl_model.epochs,
            report_train_metrics_iteration=config.common.report_train_metrics_iteration,
            report_test_metrics_iteration=config.common.report_test_metrics_iteration,
            processor=processors[0],
            target_uids=target_uids,
            batch_size=config.vfl_model.batch_size,
            model_update_dim_size=0,
            run_mlflow=config.master.run_mlflow,
            security_protocol=SecurityProtocolPaillier(),
        )

        member_ids = [f"member-{member_rank}" for member_rank in range(config.common.world_size)]

        members = [
            member_class(
                uid=member_uid,
                member_record_uids=target_uids,
                processor=processors[member_rank + 1],
                batch_size=config.vfl_model.batch_size,
                epochs=config.vfl_model.epochs,
                security_protocol=SecurityProtocolPaillier(),
            )
            for member_rank, member_uid in enumerate(member_ids)
        ]

        def local_master_main():
            logger.info("Starting thread %s" % threading.current_thread().name)
            comm = ArbiteredLocalPartyCommunicator(
                participant=master,
                world_size=config.common.world_size,
                shared_party_info=shared_party_info,
                recv_timeout=config.master.recv_timeout,
            )
            comm.run()
            logger.info("Finishing thread %s" % threading.current_thread().name)

        def local_member_main(member: PartyMember):
            logger.info("Starting thread %s" % threading.current_thread().name)
            comm = ArbiteredLocalPartyCommunicator(
                participant=member,
                world_size=config.common.world_size,
                shared_party_info=shared_party_info,
                recv_timeout=config.member.recv_timeout,
            )
            comm.run()
            logger.info("Finishing thread %s" % threading.current_thread().name)

        def local_arbiter_main():
            logger.info("Starting thread %s" % threading.current_thread().name)
            comm = ArbiteredLocalPartyCommunicator(
                participant=arbiter,
                world_size=config.common.world_size,
                shared_party_info=shared_party_info,
                recv_timeout=360.,
            )
            comm.run()
            logger.info("Finishing thread %s" % threading.current_thread().name)


        run_local_agents(
            master=master,
            members=members,
            target_master_func=local_master_main,
            target_member_func=local_member_main,
            arbiter=arbiter,
            target_arbiter_func=local_arbiter_main,

        )


if __name__ == "__main__":
    run()
