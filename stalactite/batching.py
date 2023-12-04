from typing import List

from stalactite.base import Batcher


class ListBatcher(Batcher):
    def __init__(self, uids: List[str], batch_size: int):
        self.uids = uids
        self.batch_size = batch_size

    def __iter__(self):
        return (self.uids[i: i + self.batch_size] for i in range(0, len(self.uids), self.batch_size))
