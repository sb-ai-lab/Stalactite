import random
import logging
from typing import List

import torch

from stalactite.base import DataTensor, PartyMember

logger = logging.getLogger("my_logger")


class PartyMemberImpl(PartyMember):
    def initialize(self):
        pass

    def finalize(self):
        pass

    def predict(self, batch) -> DataTensor:
        logger.debug(f"PARTY MEMBER: making predict...")
        return torch.rand(len(batch))

    def records_uids(self) -> List[str]:
        pass

    def register_records_uids(self, uids: List[str]):
        pass

    def update_predict(self, batch: List[str], upd: DataTensor) -> DataTensor:
        self.update_weights(upd)
        return self.predict(batch)

    def update_weights(self, upd: DataTensor):
        logger.debug(f"PARTY MEMBER: updating weights...")
        # todo: add batch here (uid part)? Q Nik