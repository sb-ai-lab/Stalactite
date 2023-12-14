from typing import List, Optional, Iterator

from stalactite.base import Batcher, TrainingIteration, RecordsBatch


class ListBatcher(Batcher):
    def __init__(self, epochs: int, members: List[str], uids: List[str], batch_size: int):
        self.epochs = epochs
        self.members = members
        self.uids = uids
        self.batch_size = batch_size

    def __iter__(self) -> Iterator[TrainingIteration]:
        def _iter_func():
            iter_num = 0
            previous_batch: Optional[RecordsBatch] = None
            for epoch_num in range(self.epochs):
                for i in range(0, len(self.uids), self.batch_size):
                    batch = self.uids[i: i + self.batch_size]
                    yield TrainingIteration(
                        seq_num=iter_num,
                        subiter_seq_num=0,
                        epoch=epoch_num,
                        batch=batch,
                        previous_batch=previous_batch,
                        participating_members=self.members
                    )
                    iter_num += 1
                    previous_batch = batch
        return _iter_func()
