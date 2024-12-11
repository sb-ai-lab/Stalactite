# TODO: this file is added temporary. It will be removed or significantly changed after refactoring of the preprocessors
import logging

from stalactite.base import PartyMaster, PartyMember
from stalactite.ml.arbitered.base import PartyArbiter
from stalactite.ml import (
    HonestPartyMasterLinReg,
    HonestPartyMasterLinRegConsequently,
    HonestPartyMemberLogReg,
    HonestPartyMemberResNet,
    HonestPartyMemberEfficientNet,
    HonestPartyMemberMLP,
    HonestPartyMemberLinReg,
    HonestPartyMasterLogReg,
    HonestPartyMasterResNetSplitNN,
    HonestPartyMasterEfficientNetSplitNN,
    HonestPartyMasterMLPSplitNN,
    PartyArbiterLogReg,
    ArbiteredPartyMasterLogReg,
    ArbiteredPartyMemberLogReg,
)

from stalactite.configs import VFLConfig
from stalactite.helpers import get_plugin_agent
from stalactite.utils import Role

from examples.utils.local_experiment import load_processors as load_processors_honest
from examples.utils.local_arbitered_experiment import load_processors as load_processors_arbitered
from stalactite.ml.arbitered.security_protocols.paillier_sp import (
    SecurityProtocolPaillier,
    SecurityProtocolArbiterPaillier,
)


def get_party_master(config_path: str, is_infer: bool = False) -> PartyMaster:
    config = VFLConfig.load_and_validate(config_path)
    if config.grpc_arbiter.use_arbiter:
        master_processor, processors = load_processors_arbitered(config)
        master_processor = master_processor if config.data.dataset.lower() in [
            "sbol_smm", "sbol_master_only_labels"
        ] else processors[0]
        if config.data.dataset_size == -1:
            config.data.dataset_size = len(master_processor.dataset[config.data.train_split][config.data.uids_key])
        if config.vfl_model.vfl_model_name in ['logreg']:
            master_class = ArbiteredPartyMasterLogReg
        else:
            master_class = get_plugin_agent(config.vfl_model.vfl_model_name, Role.master)
        if config.grpc_arbiter.security_protocol_params is not None:
            if config.grpc_arbiter.security_protocol_params.he_type == 'paillier':
                sp_agent = SecurityProtocolPaillier(**config.grpc_arbiter.security_protocol_params.init_params)
            else:
                raise ValueError('Only paillier HE implementation is available')
        else:
            sp_agent = None
        return master_class(
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
            do_predict=is_infer,
            do_train=not is_infer,
            do_save_model=config.vfl_model.do_save_model,
            model_path=config.vfl_model.vfl_model_path,
            seed=config.common.seed,
            device='cuda' if config.docker.use_gpu else 'cpu',
        )

    else:
        master_processor, processors = load_processors_honest(config)
        if config.data.dataset_size == -1:
            config.data.dataset_size = len(master_processor.dataset[config.data.train_split][config.data.uids_key])
        if config.vfl_model.vfl_model_name in ['logreg', 'resnet', 'efficientnet', 'mlp', 'linreg']:
            if 'logreg' in config.vfl_model.vfl_model_name:
                master_class = HonestPartyMasterLogReg
            elif "resnet" in config.vfl_model.vfl_model_name:
                master_class = HonestPartyMasterResNetSplitNN
            elif "efficientnet" in config.vfl_model.vfl_model_name:
                master_class = HonestPartyMasterEfficientNetSplitNN
            elif "mlp" in config.vfl_model.vfl_model_name:
                master_class = HonestPartyMasterMLPSplitNN
            else:
                if config.vfl_model.is_consequently:
                    master_class = HonestPartyMasterLinRegConsequently
                else:
                    master_class = HonestPartyMasterLinReg
        else:
            master_class = get_plugin_agent(config.vfl_model.vfl_model_name, Role.master)

        return master_class(
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
            do_predict=is_infer,
            do_train=not is_infer,
            model_name=config.vfl_model.vfl_model_name if
            config.vfl_model.vfl_model_name in ["resnet", "mlp", "efficientnet"] else None,
            model_params=config.master.master_model_params,
            seed=config.common.seed,
            device='cuda' if config.docker.use_gpu else 'cpu',
            do_save_model=config.vfl_model.do_save_model,
            model_path=config.vfl_model.vfl_model_path,
        )


def get_party_member(config_path: str, member_rank: int, is_infer: bool = False) -> PartyMember:
    config = VFLConfig.load_and_validate(config_path)
    if config.grpc_arbiter.use_arbiter:
        master_processor, processors = load_processors_arbitered(config)
        if config.vfl_model.vfl_model_name in ['logreg']:
            member_class = ArbiteredPartyMemberLogReg
        else:
            member_class = get_plugin_agent(config.vfl_model.vfl_model_name, Role.member)
        if config.grpc_arbiter.security_protocol_params is not None:
            if config.grpc_arbiter.security_protocol_params.he_type == 'paillier':
                sp_agent = SecurityProtocolPaillier(**config.grpc_arbiter.security_protocol_params.init_params)
            else:
                raise ValueError('Only paillier HE implementation is available')
        else:
            sp_agent = None

        return member_class(
            uid=f"member-{member_rank}",
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
            do_predict=is_infer,
            do_train=not is_infer,
            do_save_model=config.vfl_model.do_save_model,
            model_path=config.vfl_model.vfl_model_path,
            use_inner_join=False,
            seed=config.common.seed,
            device='cuda' if config.docker.use_gpu else 'cpu',
        )

    else:
        master_processor, processors = load_processors_honest(config)
        if config.vfl_model.vfl_model_name in ['logreg', 'resnet', 'efficientnet', 'mlp', 'linreg']:
            if 'logreg' in config.vfl_model.vfl_model_name:
                member_class = HonestPartyMemberLogReg
            elif "resnet" in config.vfl_model.vfl_model_name:
                member_class = HonestPartyMemberResNet
            elif "efficientnet" in config.vfl_model.vfl_model_name:
                member_class = HonestPartyMemberEfficientNet
            elif "mlp" in config.vfl_model.vfl_model_name:
                member_class = HonestPartyMemberMLP
            else:
                member_class = HonestPartyMemberLinReg
        else:
            member_class = get_plugin_agent(config.vfl_model.vfl_model_name, Role.member)

        member_ids = [f"member-{member_rank}" for member_rank in range(config.common.world_size)]
        return member_class(
            uid=f"member-{member_rank}",
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
            do_predict=is_infer,
            do_train=not is_infer,
            do_save_model=config.vfl_model.do_save_model,
            model_path=config.vfl_model.vfl_model_path,
            model_params=config.member.member_model_params,
            use_inner_join=True if member_rank == 0 else False,
            seed=config.common.seed,
            device='cuda' if config.docker.use_gpu else 'cpu',
        )


def get_party_arbiter(config_path: str, is_infer: bool = False) -> PartyArbiter:
    config = VFLConfig.load_and_validate(config_path)
    if not config.grpc_arbiter.use_arbiter:
        raise RuntimeError('Arbiter should not be called in honest setting.')
    if config.vfl_model.vfl_model_name in ['logreg']:
        arbiter_class = PartyArbiterLogReg
    else:
        arbiter_class = get_plugin_agent(config.vfl_model.vfl_model_name, Role.arbiter)
    if config.grpc_arbiter.security_protocol_params is not None:
        if config.grpc_arbiter.security_protocol_params.he_type == 'paillier':
            sp_arbiter = SecurityProtocolArbiterPaillier(**config.grpc_arbiter.security_protocol_params.init_params)
        else:
            raise ValueError('Only paillier HE implementation is available')
    else:
        sp_arbiter = None

    return arbiter_class(
        uid="arbiter",
        epochs=config.vfl_model.epochs,
        batch_size=config.vfl_model.batch_size,
        eval_batch_size=config.vfl_model.eval_batch_size,
        security_protocol=sp_arbiter,
        learning_rate=config.vfl_model.learning_rate,
        momentum=0.0,
        num_classes=config.data.num_classes,
        do_predict=is_infer,
        do_train=not is_infer,
    )
