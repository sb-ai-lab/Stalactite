import logging
from typing import List, Optional, Any, Union

import torch.nn.parameter

from stalactite.base import DataTensor, Batcher
from stalactite.batching import ListBatcher
from stalactite.ml.arbitered.base import PartyArbiter, SecurityProtocolArbiter
from stalactite.utils import Role

logger = logging.getLogger(__name__)


class PartyArbiterLogReg(PartyArbiter):
    role = Role.arbiter

    def __init__(
            self,
            uid: str,
            epochs: int,
            num_classes: int,
            batch_size: int,
            eval_batch_size: int,
            security_protocol: Optional[SecurityProtocolArbiter] = None,
            learning_rate: float = 0.01,
            momentum: float = 0.0,
            do_train: bool = True,
            do_predict: bool = False,
            **kwargs,
    ) -> None:
        self.id = uid
        self.epochs = epochs
        self.num_classes = num_classes
        self._batch_size = batch_size
        self._eval_batch_size = eval_batch_size
        self.security_protocol = security_protocol
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.do_train = do_train
        self.do_predict = do_predict

        if kwargs:
            logger.info(f'Passed extra kwargs to arbiter ({kwargs}), ignoring.')

        self.is_initialized = False
        self.is_finalized = False

        self._model_initialized = False
        if self.num_classes < 1:
            raise AttributeError('Number of classes (`num_classes`) must be a positive integer')

        self._prev_model_parameter: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None
        self._model_parameter: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None
        self._optimizer: Optional[Union[torch.optim.Optimizer, List[torch.optim.Optimizer]]] = None
        self.members: Optional[List[str]] = None
        self.master: Optional[str] = None
        self._uids_to_use: Optional[List[str]] = None
        self._uids_to_use_test: Optional[List[str]] = None

    def make_batcher(
            self,
            uids: Optional[List[str]] = None,
            party_members: Optional[List[str]] = None,
            is_infer: bool = False
    ) -> Batcher:
        if uids is None:
            uids = self._uids_to_use_test if is_infer else self._uids_to_use
        logger.info(f"Arbiter {self.id} makes a batcher for {len(uids)} uids")

        epochs = 1 if is_infer else self.epochs
        batch_size = self._eval_batch_size if is_infer else self._batch_size
        return ListBatcher(
            epochs=epochs,
            members=self.members if party_members is None else party_members,
            uids=uids,
            batch_size=batch_size,
        )

    def initialize_model_from_params(self, **model_params) -> Any:
        raise AttributeError('Arbiter is not included in the inference process.')

    def _init_optimizer(self):
        self._optimizer = [
            torch.optim.SGD([self._model_parameter[class_idx]], lr=self.learning_rate, momentum=self.momentum)
            for class_idx in range(self.num_classes)
        ]

    def _init_model_parameter(self, features_len: int, dtype: torch.float64):
        self._model_parameter = [
            torch.nn.parameter.Parameter(torch.zeros((features_len, 1), requires_grad=True, dtype=dtype))
            for _ in range(self.num_classes)
        ]
        self._init_optimizer()
        self._model_initialized = True

    def _optimizer_step(self, gradient: torch.Tensor):
        self._prev_model_parameter = []
        for class_idx in range(self.num_classes):
            self._optimizer[class_idx].zero_grad()
            self._prev_model_parameter.append(self._model_parameter[class_idx].data.clone())
            self._model_parameter[class_idx].grad = gradient[class_idx].to('cpu')  # Local communicator requirement
            self._optimizer[class_idx].step()

    def _get_delta_gradients(self) -> torch.Tensor:
        if self._prev_model_parameter is not None:
            return torch.stack(self._prev_model_parameter) - torch.stack([mp.data for mp in self._model_parameter])
        else:
            raise ValueError(f"No previous steps were performed.")

    def calculate_updates(self, gradients: dict) -> dict[str, DataTensor]:
        logger.info(f'Arbiter {self.id} calculates updates for {len(gradients)} agents')
        members = [key for key in gradients.keys() if key != self.master]

        try:
            master_gradient = gradients[self.master]
            members_gradients = [gradients[member] for member in members]
        except KeyError:
            raise RuntimeError(f'Master did not pass the gradient or was not initialized ({self.master})')

        if self.security_protocol is not None:
            master_gradient = self.security_protocol.decrypt(master_gradient)

        size_list = [master_gradient.size()[1]]
        gradient = [grad.squeeze() for grad in master_gradient]

        for member_gradient in members_gradients:
            if self.security_protocol is not None:
                member_gradient = self.security_protocol.decrypt(member_gradient)

            size_list.append(member_gradient.size()[1])
            for class_idx in range(self.num_classes):
                gradient[class_idx] = torch.hstack((gradient[class_idx], member_gradient[class_idx].squeeze()))

        if not self._model_initialized:
            self._init_model_parameter(sum(size_list), dtype=gradient[0].dtype)

        self._optimizer_step(torch.stack([grad.unsqueeze(1) for grad in gradient]))
        delta_gradients = self._get_delta_gradients()

        splitted_grads = torch.tensor_split(delta_gradients, torch.cumsum(torch.tensor(size_list), 0)[:-1], dim=1)
        deltas = {agent: splitted_grads[i].clone().detach() for i, agent in enumerate([self.master] + members)}
        logger.debug(f'Arbiter {self.id} has calculated updates')
        return deltas

    def initialize(self, is_infer: bool = False):
        logger.info(f"Arbiter {self.id}: initializing")
        if self.security_protocol is not None:
            self.security_protocol.generate_keys()
        self.is_initialized = True
        self.is_finalized = False
        logger.info(f"Arbiter {self.id}: has been initialized")

    def finalize(self, is_infer: bool = False):
        logger.info(f"Arbiter {self.id}: finalizing")
        self.is_finalized = True
        logger.info(f"Arbiter {self.id} has finalized")

    def register_records_uids(self, uids: List[str], is_infer: bool = False):
        """ Register unique identifiers to be used.

        :param uids: List of unique identifiers.
        :return: None
        """
        logger.info(f"Arbiter {self.id}: registering {len(uids)} uids to be used.")
        if is_infer:
            self._uids_to_use_test = uids
        else:
            self._uids_to_use = uids
