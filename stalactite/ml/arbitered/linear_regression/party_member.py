from typing import List

import logging

from stalactite.base import Batcher, PartyCommunicator, DataTensor, RecordsBatch
from stalactite.ml.arbitered.base import ArbiteredPartyMember

logger = logging.getLogger(__name__)

logger.setLevel(logging.DEBUG)
sh = logging.StreamHandler()
sh.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
logger.addHandler(sh)
class ArbiteredPartyMemberLinReg(ArbiteredPartyMember):
    _dataset

    def _check_if_ready(self):

        ...

    def predict_partial(self, uids: RecordsBatch, subiter_seq_num: int) -> DataTensor:
        logger.info("Member %s: predicting. Batch size: %s" % (self.id, len(uids)))
        self._check_if_ready()
        X = self._dataset[self._data_params.train_split][self._data_params.features_key][[int(x) for x in uids]]
        predictions = self._model.predict(X)
        logger.info("Member %s: made predictions." % self.id)
        return predictions

    def compute_gradient(self, aggregated_predictions_diff: DataTensor, uids: List[str]) -> DataTensor:
        pass

    @property
    def batcher(self) -> Batcher:
        pass

    def records_uids(self) -> List[str]:
        pass

    def register_records_uids(self, uids: List[str]):
        pass

    def update_weights(self, uids: RecordsBatch, upd: DataTensor):
        pass

    def initialize(self):
        pass

    def finalize(self):
        pass