import argparse
import logging
from functools import wraps
import time
from typing import Any, Generator, Callable

import numpy as np
import torch

from experiments.src.utils.func_utils import save_data, load_data, BatchedData
from experiments.src.utils.helpers import (
    ClientTask,
    format_important_logging,
    Serialization,
    generate_data,
    safetensor_collect_results_unary,
    safetensor_collect_results_stream,
    prototensor_collect_results_unary,
    prototensor_collect_results_stream,
)

parser = argparse.ArgumentParser()

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
sh = logging.StreamHandler()
sh.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(sh)


def log_timing(logger: Callable = print):
    def _decorator(func):
        @wraps(func)
        def _wrap(*args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)
            total_time = round(time.time() - start, 4)
            logging_str = f'Function {func.__name__} finished in {total_time} sec'
            logger(logging_str)
            return result

        return _wrap

    return _decorator


class ExperimentalData:
    def __init__(
            self, num_rows: int, num_columns: int, dtype: torch.dtype, batch_size: int, serialization: Serialization
    ):
        self.num_rows = num_rows
        self.num_columns = num_columns
        self.dtype = dtype
        self.batch_size = batch_size
        self.serialization = serialization

    @generate_data.time()
    def generate_data(self) -> torch.Tensor:
        return torch.rand(self.num_rows, self.num_columns, dtype=self.dtype)

    def batch_generate_data(self) -> Generator[BatchedData, None, None]:
        total_batches = int(np.ceil(self.num_rows / self.batch_size))
        for batch in range(total_batches):
            with generate_data.time():
                yield BatchedData(
                    data=torch.rand(self.batch_size, self.num_columns, dtype=self.dtype),
                    batch=batch,
                    total_batches=total_batches,
                )


class Task:
    def __init__(self, task_type: ClientTask, data: ExperimentalData | None = None):
        self.task_type = task_type
        self.data = data

        if self.task_type != ClientTask.finish and self.data is None:
            raise ValueError('Data is required if task type is not ClientTask.finish')

    @property
    def type(self) -> ClientTask:
        return self.task_type

    def _exchange_messages_generator(self, client_name: str, client_iteration: int):
        for batch in self.data.batch_generate_data():
            yield save_data(
                tensor=batch.data,
                client_id=client_name,
                client_iteration=client_iteration,
                serialization=self.data.serialization,
                batch=batch.batch,
                total_batches=batch.total_batches,
            )

    def start_rpc(
            self,
            stub,
            client_name: str,
            client_iteration: int,
            timeout: float = 3600,
    ):
        if self.task_type == ClientTask.exchange_tensor:
            return stub.ExchangeBinarizedDataUnaryUnary(
                save_data(
                    tensor=self.data.generate_data(),
                    client_id=client_name,
                    client_iteration=client_iteration,
                    serialization=self.data.serialization,
                ),
                timeout=timeout)
        elif self.task_type == ClientTask.exchange_array:
            return stub.ExchangeNumpyDataUnaryUnary(
                save_data(
                    tensor=self.data.generate_data(),
                    client_id=client_name,
                    client_iteration=client_iteration,
                    serialization=self.data.serialization,
                ),
                timeout=timeout
            )
        elif self.task_type == ClientTask.batched_exchange_tensor:
            return stub.ExchangeBinarizedDataStreamStream(
                self._exchange_messages_generator(client_name, client_iteration),
                wait_for_ready=True
            )
        elif self.task_type == ClientTask.batched_exchange_array:
            return stub.ExchangeNumpyDataStreamStream(
                self._exchange_messages_generator(client_name, client_iteration),
                wait_for_ready=True
            )
        else:
            logger.warning(f"Got task type {self.task_type}. Skipping")
            return

    async def collect_results_rpc(self, future: Any):
        result = torch.tensor([])
        if self.task_type in (ClientTask.exchange_tensor, ClientTask.exchange_array):
            start = time.time()
            time_collection_method = safetensor_collect_results_unary if self.task_type == ClientTask.exchange_tensor \
                else prototensor_collect_results_unary
            with time_collection_method.time():
                data = await future
                breakpoint = time.time()
                serialization = Serialization.safetensors if self.task_type == ClientTask.exchange_tensor else Serialization.protobuf
                result = load_data(data, serialization=serialization)
            end = time.time()
            total_time = end - start
            awaiting_time = breakpoint - start
            deserialization_time = end - breakpoint
            logger.info(format_important_logging(
                f"Result got in {round(total_time, 4)} sec: \n"
                f" - coro awaited for {round(awaiting_time, 4)} sec;\n"
                f" - deserialization time {round(deserialization_time, 4)}"
            ))
        elif self.task_type in (ClientTask.batched_exchange_tensor, ClientTask.batched_exchange_array):
            start = time.time()
            serialization = Serialization.safetensors if self.task_type == ClientTask.batched_exchange_tensor \
                else Serialization.protobuf
            time_collection_method = safetensor_collect_results_stream \
                if self.task_type == ClientTask.batched_exchange_tensor else prototensor_collect_results_stream
            with time_collection_method.time():
                async for batch in future:
                    data_batch = load_data(batch, serialization=serialization)
                    result = torch.cat([result, data_batch])
            end = time.time()
            logger.info(format_important_logging(
                f"Result got in {round(end - start, 4)} sec: \n"
            ))
        else:
            return
        return result
