import random
from typing import Iterator, List, Optional

from stalactite.base import Batcher, RecordsBatch, TrainingIteration


class ListBatcher(Batcher):
    def __init__(
            self, epochs: int, members: Optional[List[str]], uids: List[str], batch_size: int, shuffle: bool = False
    ):
        self.epochs = epochs
        self.members = members
        self.uids = uids
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self) -> Iterator[TrainingIteration]:
        def _iter_func():
            iter_num = 0
            previous_batch: Optional[RecordsBatch] = None
            for epoch_num in range(self.epochs):
                if self.shuffle:
                    random.shuffle(self.uids)
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
                        last_batch=False,
                    )
                    iter_num += 1
                    iter_in_batch += 1
                    previous_batch = batch
            yield TrainingIteration(
                seq_num=iter_num - 1,
                subiter_seq_num=iter_in_batch - 1,
                epoch=epoch_num,
                batch=batch,
                previous_batch=None,
                participating_members=self.members,
                last_batch=True
            )

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
                            last_batch=False
                        )
                    iter_num += 1
                    iter_in_batch += 1
                    previous_batch = batch
            for member in self.members:
                yield TrainingIteration(
                    seq_num=iter_num-1,
                    subiter_seq_num=iter_in_batch-1,
                    epoch=epoch_num,
                    batch=batch,
                    previous_batch=None,
                    participating_members=[member],
                    last_batch=True,
                )
        return _iter_func()
