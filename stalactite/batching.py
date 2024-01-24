from typing import Iterator, List, Optional

from stalactite.base import Batcher, RecordsBatch, TrainingIteration


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
                iter_in_batch = 0
                for i in range(0, len(self.uids), self.batch_size):
                    batch = self.uids[i : i + self.batch_size]
                    yield TrainingIteration(
                        seq_num=iter_num,
                        subiter_seq_num=iter_in_batch,
                        epoch=epoch_num,
                        batch=batch,
                        previous_batch=previous_batch,
                        participating_members=self.members,
                    )
                    iter_num += 1
                    iter_in_batch += 1
                    previous_batch = batch

        return _iter_func()


class ConsecutiveListBatcher(ListBatcher):
    def __iter__(self) -> Iterator[TrainingIteration]:
        def _iter_func():
            iter_num = 0
            previous_batch: Optional[RecordsBatch] = None
            for epoch_num in range(self.epochs):
                iter_in_batch = 0
                for i in range(0, len(self.uids), self.batch_size):
                    batch = self.uids[i : i + self.batch_size]
                    for member in self.members:
                        yield TrainingIteration(
                            seq_num=iter_num,
                            subiter_seq_num=iter_in_batch,
                            epoch=epoch_num,
                            batch=batch,
                            previous_batch=previous_batch,
                            participating_members=[member],
                        )
                    iter_num += 1
                    iter_in_batch += 1
                    previous_batch = batch

        return _iter_func()
