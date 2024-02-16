# TODO: this file is added temporary. It will be removed or significantly changed after refactoring of the preprocessors

from stalactite.ml import (
    HonestPartyMasterLinRegConsequently,
    HonestPartyMasterLinReg,
    HonestPartyMemberLogReg,
    HonestPartyMemberLinReg,
    HonestPartyMasterLogReg
)


from stalactite.configs import VFLConfig

from examples.utils.local_experiment import load_processors


def get_party_master(config_path: str):
    config = VFLConfig.load_and_validate(config_path)
    processors = load_processors(config)

    target_uids = [str(i) for i in range(config.data.dataset_size)]
    if 'logreg' in config.common.vfl_model_name:
        master_class = HonestPartyMasterLogReg
    else:
        if config.common.is_consequently:
            master_class = HonestPartyMasterLinRegConsequently
        else:
            master_class = HonestPartyMasterLinReg
    return master_class(
        uid="master",
        epochs=config.vfl_model.epochs,
        report_train_metrics_iteration=config.common.report_train_metrics_iteration,
        report_test_metrics_iteration=config.common.report_test_metrics_iteration,
        processor=processors[0],
        target_uids=target_uids,
        batch_size=config.vfl_model.batch_size,
        model_update_dim_size=0,
        run_mlflow=config.master.run_mlflow,
    )


def get_party_member(config_path: str, member_rank: int):
    config = VFLConfig.load_and_validate(config_path)
    processors = load_processors(config)
    target_uids = [str(i) for i in range(config.data.dataset_size)]
    if 'logreg' in config.common.vfl_model_name:
        member_class = HonestPartyMemberLogReg
    else:
        member_class = HonestPartyMemberLinReg
    return member_class(
        uid=f"member-{member_rank}",
        member_record_uids=target_uids,
        model_name=config.vfl_model.vfl_model_name,
        processor=processors[member_rank],
        batch_size=config.vfl_model.batch_size,
        epochs=config.vfl_model.epochs,
        report_train_metrics_iteration=config.common.report_train_metrics_iteration,
        report_test_metrics_iteration=config.common.report_test_metrics_iteration,
    )
