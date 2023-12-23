import enum


class _Method(str, enum.Enum):
    service_return_answer = 'service_return_answer'
    service_heartbeat = 'service_heartbeat'

    records_uids = 'records_uids'
    register_records_uids = 'register_records_uids'

    initialize = 'initialize'
    finalize = 'finalize'

    update_weights = 'update_weights'
    predict = 'predict'
    update_predict = 'update_predict'


METHOD_VALUES = {
    _Method.records_uids: 'other_kwargs',
    _Method.register_records_uids: 'other_kwargs',
    _Method.initialize: 'other_kwargs',
    _Method.finalize: 'other_kwargs',
    _Method.update_weights: 'other_kwargs',
    _Method.predict: 'tensor_kwargs',
    _Method.update_predict: 'tensor_kwargs',
}


class ParticipantType(enum.Enum):
    master = 'master'
    member = 'member'
