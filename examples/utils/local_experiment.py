import os
import logging
from pathlib import Path
import threading
from typing import Optional
import datasets

from stalactite.ml import (
    HonestPartyMasterLinRegConsequently,
    HonestPartyMasterLinReg,
    HonestPartyMemberLogReg,
    HonestPartyMemberLinReg,
    HonestPartyMasterLogReg,
    HonestPartyMemberResNet,
    HonestPartyMasterResNetSplitNN,
    HonestPartyMemberEfficientNet,
    HonestPartyMasterEfficientNetSplitNN,
    HonestPartyMasterMLPSplitNN,
    HonestPartyMemberMLP
)
from stalactite.data_preprocessors import ImagePreprocessor, TabularPreprocessor, ImagePreprocessorEff
from stalactite.communications.local import LocalMasterPartyCommunicator, LocalMemberPartyCommunicator
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

        binary = False if config.vfl_model.vfl_model_name in ["efficientnet", "logreg"] else True

        if len(os.listdir(config.data.host_path_data_dir)) == 0:
            load_mnist(
                save_path=Path(config.data.host_path_data_dir), parts_num=config.common.world_size, binary=binary
            )

        dataset = {}
        for m in range(config.common.world_size):
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
            load_sbol(os.path.dirname(config.data.host_path_data_dir), parts_num=2, sample=config.data.sample,
                      seed=config.common.seed, use_smm=True)

        for m in range(config.common.world_size):
            dataset[m] = datasets.load_from_disk(
                os.path.join(f"{config.data.host_path_data_dir}/part_{m}")
            )
        processors = [
            TabularPreprocessor(dataset=dataset[i], member_id=i, params=config) for i, v in dataset.items()
        ]
        master_processor = TabularPreprocessor(dataset=datasets.load_from_disk(
                os.path.join(f"{config.data.host_path_data_dir}/master_part")
            ), member_id=-1, params=config, is_master=True)

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
    if config.data.dataset_size == -1:
        config.data.dataset_size = len(master_processor.dataset[config.data.train_split][config.data.uids_key])

    with reporting(config):
        shared_party_info = dict()
        if 'logreg' in config.vfl_model.vfl_model_name:
            master_class = HonestPartyMasterLogReg
            member_class = HonestPartyMemberLogReg
        elif "resnet" in config.vfl_model.vfl_model_name:
            master_class = HonestPartyMasterResNetSplitNN
            member_class = HonestPartyMemberResNet
        elif "efficientnet" in config.vfl_model.vfl_model_name:
            master_class = HonestPartyMasterEfficientNetSplitNN
            member_class = HonestPartyMemberEfficientNet
        elif "mlp" in config.vfl_model.vfl_model_name:
            master_class = HonestPartyMasterMLPSplitNN
            member_class = HonestPartyMemberMLP
        else:
            member_class = HonestPartyMemberLinReg
            if config.vfl_model.is_consequently:
                master_class = HonestPartyMasterLinRegConsequently
            else:
                master_class = HonestPartyMasterLinReg
        master = master_class(
            uid="master",
            epochs=config.vfl_model.epochs,
            report_train_metrics_iteration=config.common.report_train_metrics_iteration,
            report_test_metrics_iteration=config.common.report_test_metrics_iteration,
            processor=master_processor,
            target_uids=master_processor.dataset[config.data.train_split][config.data.uids_key][:config.data.dataset_size],
            inference_target_uids=master_processor.dataset[config.data.test_split][config.data.uids_key],
            batch_size=config.vfl_model.batch_size,
            eval_batch_size=config.vfl_model.eval_batch_size,
            model_update_dim_size=0,
            run_mlflow=config.master.run_mlflow,
            do_train=config.vfl_model.do_train,
            do_predict=config.vfl_model.do_predict,
            model_name=config.vfl_model.vfl_model_name if
            config.vfl_model.vfl_model_name in ["resnet", "mlp", "efficientnet"] else None,
            model_params=config.master.master_model_params,
            seed=config.common.seed
        )

        member_ids = [f"member-{member_rank}" for member_rank in range(config.common.world_size)]

        members = [
            member_class(
                uid=member_uid,
                member_record_uids=processors[member_rank].dataset[config.data.train_split][config.data.uids_key],
                member_inference_record_uids=processors[member_rank].dataset[config.data.test_split][config.data.uids_key],
                model_name=config.vfl_model.vfl_model_name,
                processor=processors[member_rank],
                batch_size=config.vfl_model.batch_size,
                eval_batch_size=config.vfl_model.eval_batch_size,
                epochs=config.vfl_model.epochs,
                report_train_metrics_iteration=config.common.report_train_metrics_iteration,
                report_test_metrics_iteration=config.common.report_test_metrics_iteration,
                is_consequently=config.vfl_model.is_consequently,
                members=member_ids if config.vfl_model.is_consequently else None,
                do_train=config.vfl_model.do_train,
                do_predict=config.vfl_model.do_predict,
                do_save_model=config.vfl_model.do_save_model,
                model_path=config.vfl_model.vfl_model_path,
                model_params=config.member.member_model_params,
                use_inner_join=True if member_rank == 0 else False,
                seed=config.common.seed

            )
            for member_rank, member_uid in enumerate(member_ids)
        ]

        def local_master_main():
            logger.info("Starting thread %s" % threading.current_thread().name)
            comm = LocalMasterPartyCommunicator(
                participant=master,
                world_size=config.common.world_size,
                shared_party_info=shared_party_info,
                recv_timeout=config.master.recv_timeout,
            )
            comm.run()
            logger.info("Finishing thread %s" % threading.current_thread().name)

        def local_member_main(member: PartyMember):
            logger.info("Starting thread %s" % threading.current_thread().name)
            comm = LocalMemberPartyCommunicator(
                participant=member,
                world_size=config.common.world_size,
                shared_party_info=shared_party_info,
                recv_timeout=config.member.recv_timeout,
            )
            comm.run()
            logger.info("Finishing thread %s" % threading.current_thread().name)

        run_local_agents(
            master=master, members=members, target_master_func=local_master_main, target_member_func=local_member_main
        )


if __name__ == "__main__":
    run()
