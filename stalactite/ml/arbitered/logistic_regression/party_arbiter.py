import logging
from typing import List, Optional

import torch.nn.parameter

from stalactite.base import DataTensor, Batcher
from stalactite.batching import ListBatcher
from stalactite.ml.arbitered.base import PartyArbiter, SecurityProtocolArbiter

logger = logging.getLogger(__name__)

logger.setLevel(logging.DEBUG)
sh = logging.StreamHandler()
sh.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
logger.addHandler(sh)


class PartyArbiterLogReg(PartyArbiter):
    _uids_to_use: List[str]
    _prev_model_parameter: Optional[torch.Tensor] = None
    _model_parameter: Optional[torch.Tensor] = None
    _optimizer: Optional[torch.optim.Optimizer] = None
    members: List[str]
    master: str

    def __init__(
            self,
            uid: str,
            epochs: int,
            batch_size: int,
            security_protocol: SecurityProtocolArbiter,
            learning_rate: float = 0.01,
            momentum: float = 0.0,
    ) -> None:
        self.id = uid
        self.epochs = epochs
        self.batch_size = batch_size
        self.security_protocol = security_protocol
        self.learning_rate = learning_rate
        self.momentum = momentum

        self.is_initialized = False
        self.is_finalized = False
        self._batcher = None

        self._model_initialized = False

    def _init_optimizer(self):
        self._optimizer = torch.optim.SGD([self._model_parameter], lr=self.learning_rate, momentum=self.momentum)

    def _init_model_parameter(self, features_len: int, dtype: torch.float64):
        # self._model_parameter = torch.nn.Linear(features_len, 1, bias=False, device=None, dtype=dtype)
        # init_weights = 0.005
        # if init_weights is not None:
        #     self._model_parameter.weight.data = torch.full((1, features_len), init_weights, requires_grad=True)
        # # self._model_parameter.weight.data = torch.zeros((1, features_len), requires_grad=True, dtype=dtype)


        self._model_parameter = torch.nn.parameter.Parameter(
            torch.zeros((features_len, 1), requires_grad=True, dtype=dtype),
        )
        self._init_optimizer()
        self._model_initialized = True

    def _optimizer_step(self, gradient: torch.Tensor):
        self._optimizer.zero_grad()
        self._prev_model_parameter = self._model_parameter.data.clone()
        self._model_parameter.grad = gradient
        self._optimizer.step()


    def _get_delta_gradients(self) -> torch.Tensor:
        if self._prev_model_parameter is not None:
            return self._prev_model_parameter - self._model_parameter.data


        else:
            raise ValueError(f"No previous steps were performed.")

    @property
    def batcher(self) -> Batcher:
        if self._batcher is None:
            if self._uids_to_use is None:
                raise RuntimeError("Cannot create make_batcher, you must `register_records_uids` first.")
            self._batcher = ListBatcher(
                epochs=self.epochs,
                members=self.members,
                uids=self._uids_to_use,
                batch_size=self.batch_size
            )
        else:
            logger.info("Member %s: using created make_batcher" % (self.id))
        return self._batcher

    def calculate_updates(self, gradients: dict) -> dict[str, DataTensor]:
        members = [key for key in gradients.keys() if key != self.master]

        try:
            master_gradient_enc = gradients[self.master]
            members_gradients_enc = [gradients[member] for member in members]
        except KeyError:
            raise RuntimeError(f'Master did not pass the gradient or was not initialized ({self.master})')

        # TODO decrypt
        master_gradient = self.security_protocol.decrypt(master_gradient_enc)
        size_list = [master_gradient.size()[0]]

        gradient = master_gradient.squeeze()

        for member_gradient_enc in members_gradients_enc:
            # TODO decrypt
            member_gradient = self.security_protocol.decrypt(member_gradient_enc)

            size_list.append(member_gradient.size()[0])
            gradient = torch.hstack((gradient, member_gradient.squeeze()))

        if not self._model_initialized:
            self._init_model_parameter(sum(size_list), dtype=gradient.dtype)

        self._optimizer_step(gradient.unsqueeze(1))
        delta_gradients = self._get_delta_gradients()

        splitted_grads = torch.tensor_split(delta_gradients, torch.cumsum(torch.tensor(size_list), 0)[:-1], dim=0)

        deltas = {agent: splitted_grads[i] for i, agent in enumerate([self.master] + members)}

        return deltas

    def initialize(self):
        self.security_protocol.generate_keys()
        self.is_initialized = True

    def finalize(self):
        self.is_finalized = True

    def register_records_uids(self, uids: List[str]):
        """ Register unique identifiers to be used.

        :param uids: List of unique identifiers.
        :return: None
        """
        logger.info("Member %s: registering %s uids to be used." % (self.id, len(uids)))
        self._uids_to_use = uids
