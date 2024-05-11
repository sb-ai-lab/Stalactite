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
from stalactite.ml.arbitered.security_protocols.paillier_sp import SecurityProtocolPaillier, \
    SecurityProtocolArbiterPaillier
from stalactite.data_preprocessors import ImagePreprocessor, TabularPreprocessor, ImagePreprocessorEff
from stalactite.communications.local import ArbiteredLocalPartyCommunicator
from stalactite.base import PartyMember
from stalactite.configs import VFLConfig
from examples.utils.prepare_mnist import load_data as load_mnist
from examples.utils.prepare_sbol import load_data as load_sbol
from stalactite.helpers import reporting, run_local_agents
from stalactite.utils import seed_all

logging.basicConfig(level=logging.DEBUG)
logging.getLogger("urllib3").setLevel(logging.CRITICAL)
logging.getLogger('PIL').setLevel(logging.ERROR)
logging.getLogger('fsspec').setLevel(logging.ERROR)
logging.getLogger('git').setLevel(logging.ERROR)
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
        for m in range(config.common.world_size + 1):
            dataset[m] = datasets.load_from_disk(
                os.path.join(f"{config.data.host_path_data_dir}/part_{m}")
            )

        image_preprocessor = ImagePreprocessorEff \
            if config.vfl_model.vfl_model_name == "efficientnet" else ImagePreprocessor

        processors = [
            image_preprocessor(dataset=dataset[i], member_id=i, params=config) for i, v in dataset.items()
        ]

        master_processor = image_preprocessor(dataset=datasets.load_from_disk(
            os.path.join(f"{config.data.host_path_data_dir}/master_part")
        ), member_id=-1, params=config, is_master=True)


    elif config.data.dataset.lower() == "sbol_smm":

        dataset = {}
        if len(os.listdir(config.data.host_path_data_dir)) == 0:
            load_sbol(os.path.dirname(config.data.host_path_data_dir), parts_num=2, sample=config.data.dataset_size,
                      seed=config.common.seed, use_smm=True)

        for m in range(1, config.common.world_size + 1):
            dataset[m] = datasets.load_from_disk(
                os.path.join(f"{config.data.host_path_data_dir}/part_{m}")
            )
        processors = [
            TabularPreprocessor(dataset=dataset[i], member_id=i, params=config) for i, v in dataset.items()
        ]
        master_processor = TabularPreprocessor(master_has_features=True, dataset=datasets.load_from_disk(
            os.path.join(f"{config.data.host_path_data_dir}/master_part_arbiter"),
        ), member_id=0, params=config, is_master=True)

    else:
        raise ValueError(f"Unknown dataset: {config.data.dataset}, choose one from ['mnist', 'multilabel']")

    return master_processor, processors


def run(config_path: Optional[str] = None):
    if config_path is None:
        config_path = os.environ.get(
            'SINGLE_MEMBER_CONFIG_PATH',
            os.path.join(Path(__file__).parent.parent, 'configs/linreg-mnist-local.yml')
        )

    config = VFLConfig.load_and_validate(config_path)
    seed_all(config.common.seed)
    master_processor, processors = load_processors(config)

    with reporting(config):

        shared_party_info = dict()
        master_class = ArbiteredPartyMasterLogReg
        member_class = ArbiteredPartyMemberLogReg
        arbiter_class = PartyArbiterLogReg
        if config.grpc_arbiter.security_protocol_params is not None:
            if config.grpc_arbiter.security_protocol_params.he_type == 'paillier':
                sp_arbiter = SecurityProtocolArbiterPaillier(**config.grpc_arbiter.security_protocol_params.init_params)
                sp_agent = SecurityProtocolPaillier(**config.grpc_arbiter.security_protocol_params.init_params)
            else:
                raise ValueError('Only paillier HE implementation is available')
        else:
            sp_arbiter, sp_agent = None, None

        arbiter = arbiter_class(
            uid="arbiter",
            epochs=config.vfl_model.epochs,
            batch_size=config.vfl_model.batch_size,
            eval_batch_size=config.vfl_model.eval_batch_size,
            security_protocol=sp_arbiter,
            learning_rate=config.vfl_model.learning_rate,
            momentum=0.0,
            num_classes=config.data.num_classes,
            do_predict=config.vfl_model.do_predict,
            do_train=config.vfl_model.do_train,
        )
        master_processor = master_processor if config.data.dataset.lower() == "sbol_smm" else processors[0]

        master = master_class(
            uid="master",
            epochs=config.vfl_model.epochs,
            report_train_metrics_iteration=config.common.report_train_metrics_iteration,
            report_test_metrics_iteration=config.common.report_test_metrics_iteration,
            processor=master_processor,
            target_uids=master_processor.dataset[config.data.train_split][config.data.uids_key][
                        :config.data.dataset_size],
            inference_target_uids=master_processor.dataset[config.data.test_split][config.data.uids_key],
            batch_size=config.vfl_model.batch_size,
            eval_batch_size=config.vfl_model.eval_batch_size,
            model_update_dim_size=0,
            run_mlflow=config.master.run_mlflow,
            num_classes=config.data.num_classes,
            security_protocol=sp_agent,
            do_predict=config.vfl_model.do_predict,
            do_train=config.vfl_model.do_train,
            do_save_model=config.vfl_model.do_save_model,
            model_path=config.vfl_model.vfl_model_path,
        )

        member_ids = [f"member-{member_rank}" for member_rank in range(config.common.world_size)]

        members = [
            member_class(
                uid=member_uid,
                member_record_uids=processors[member_rank].dataset[config.data.train_split][config.data.uids_key],
                member_inference_record_uids=processors[member_rank].dataset[config.data.test_split][
                    config.data.uids_key],
                processor=processors[member_rank],
                batch_size=config.vfl_model.batch_size,
                eval_batch_size=config.vfl_model.eval_batch_size,
                epochs=config.vfl_model.epochs,
                report_train_metrics_iteration=config.common.report_train_metrics_iteration,
                report_test_metrics_iteration=config.common.report_test_metrics_iteration,
                num_classes=config.data.num_classes,
                security_protocol=sp_agent,
                do_predict=config.vfl_model.do_predict,
                do_train=config.vfl_model.do_train,
                do_save_model=config.vfl_model.do_save_model,
                model_path=config.vfl_model.vfl_model_path,
                use_inner_join=False
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
                recv_timeout=3600.,
            )
            comm.run()
            logger.info("Finishing thread %s" % threading.current_thread().name)

        def local_arbiter_main():
            logger.info("Starting thread %s" % threading.current_thread().name)
            comm = ArbiteredLocalPartyCommunicator(
                participant=arbiter,
                world_size=config.common.world_size,
                shared_party_info=shared_party_info,
                recv_timeout=config.grpc_arbiter.recv_timeout,
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
