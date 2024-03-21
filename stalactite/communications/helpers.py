import enum
from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class MethodKwargs:
    """Data class holding keyword arguments for called method."""

    tensor_kwargs: dict[str, torch.Tensor] = field(default_factory=dict)
    other_kwargs: dict[str, Any] = field(default_factory=dict)


class Method(str, enum.Enum):
    service_return_answer = "service_return_answer"
    service_heartbeat = "service_heartbeat"

    records_uids = "records_uids"
    register_records_uids = "register_records_uids"

    initialize = "initialize"
    finalize = "finalize"

    update_weights = "update_weights"
    predict = "predict"
    update_predict = "update_predict"

    fillna = "fillna"

    get_public_key = "get_public_key"
    predict_partial = "predict_partial"
    compute_gradient = "compute_gradient"
    calculate_updates = "calculate_updates"


METHOD_VALUES = {
    Method.records_uids: "other_kwargs",
    Method.register_records_uids: "other_kwargs",
    Method.initialize: "other_kwargs",
    Method.finalize: "other_kwargs",
    Method.update_weights: "other_kwargs",
    Method.predict: "tensor_kwargs",
    Method.get_public_key: "other_kwargs",
    Method.predict_partial: "other_kwargs",
    Method.compute_gradient: "other_kwargs",
    Method.calculate_updates: "tensor_kwargs",
}


class ParticipantType(enum.Enum):
    master = "master"
    member = "member"
    arbiter = "arbiter"
