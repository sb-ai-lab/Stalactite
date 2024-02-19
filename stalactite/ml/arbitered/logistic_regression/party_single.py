from typing import List, Optional

import datasets
import mlflow
import numpy as np
import torch
from sklearn import metrics
from sklearn.metrics import roc_auc_score

from stalactite.base import PartyAgent, Batcher, DataTensor
from stalactite.batching import ListBatcher
from stalactite.configs import VFLModelConfig
from stalactite.metrics import ComputeAccuracy_numpy


class ArbiteredPartySingle(PartyAgent):
    def __init__(
            self,
            uid: str,
            epochs: int,
            batch_size: int,
            processor=None,
            learning_rate: float = 0.2,
            momentum: float = 0.,
            run_mlflow: bool = False
    ) -> None:
        """ Initialize ArbiteredPartyMasterLinReg.

        :param uid: Unique identifier for the party master.
        :param epochs: Number of training epochs.
        :param report_train_metrics_iteration: Number of iterations between reporting metrics on the train dataset.
        :param report_test_metrics_iteration: Number of iterations between reporting metrics on the test dataset.
        :param target_uids: List of unique identifiers for target dataset rows.
        :param batch_size: Size of the training batch.
        :param model_update_dim_size: Dimension size for model updates.
        :param processor: Optional data processor.
        :param run_mlflow: Flag indicating whether to use MlFlow for logging.

        :return: None
        """
        self.id = uid
        self.epochs = epochs
        self.is_initialized = False
        self.is_finalized = False
        self._batch_size = batch_size
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.processor = processor
        self.iteration_counter = 0
        self.party_predictions = dict()
        self.updates = dict()
        self.run_mlflow = run_mlflow

    _dataset: datasets.DatasetDict
    _data_params: VFLModelConfig

    def make_batcher(self, uids: List[str]) -> Batcher:
        batcher = ListBatcher(
            epochs=self.epochs,
            members=None,
            uids=uids,
            batch_size=self._batch_size
        )
        self._batcher = batcher
        return batcher

    def synchronize_uids(self):
        return [str(x) for x in range(self.target.shape[0])]

    def finalize(self):
        pass

    def _optimizer_step(self, gradient: torch.Tensor):
        # self._optimizer.zero_grad()

        self._prev_model_parameter = self._model_parameter.weight.data.clone()
        self._model_parameter.weight.grad = gradient.T
        self._optimizer.step()
        print(
            'PARAM',
            torch.sum(self._model_parameter.weight.data),
            torch.sum(self._prev_model_parameter),
            torch.sum(gradient.T),
            torch.sum(torch.where(gradient.T < 0, 0, 1)),
            torch.sum(torch.where(gradient.T > 0, 0, 1))
        )


    def _get_delta_gradients(self) -> torch.Tensor:
        if self._prev_model_parameter is not None:
            return self._prev_model_parameter - self._model_parameter.weight.data
        else:
            raise ValueError(f"No previous steps were performed.")

    def _init_optimizer(self):
        self._optimizer = torch.optim.SGD(self._model_parameter.parameters(), lr=self.learning_rate, momentum=self.momentum)

    def initialize(self):
        dataset = self.processor.fit_transform()
        self._dataset = dataset

        self.target = dataset[self.processor.data_params.train_split][self.processor.data_params.label_key][:,
                      6].unsqueeze(1)
        self.test_target = dataset[self.processor.data_params.test_split][self.processor.data_params.label_key][:,
                           6].unsqueeze(1)

        self.target = torch.where(self.target == 0., -1., 1.)
        self.test_target = torch.where(self.test_target == 0., -1., 1.)

        unique, counts = np.unique(self.target, return_counts=True)
        self._pos_weight = counts[0] / counts[1]

        self.alpha = 0.001 # l2 reg

        self._data_params = self.processor.data_params
        self._common_params = self.processor.common_params

        self._model_parameter = torch.nn.Linear(
            self._dataset[self._data_params.train_split][self._data_params.features_key].shape[1], 1, bias=False,
            device=None)
        init_weights = 0.005
        if init_weights is not None:
            self._model_parameter.weight.data = torch.full(
                (1, self._dataset[self._data_params.train_split][self._data_params.features_key].shape[1]),
                init_weights, requires_grad=True)
        # self._model_parameter.weight.data = torch.zeros((1, features_len), requires_grad=True, dtype=dtype)
        self._init_optimizer()
        self._model_initialized = True

    def run(self, party) -> None:
        """ Run centralized experiment.

        :return: None
        """
        self.initialize()
        uids = self.synchronize_uids()
        self.register_uids(uids)
        self.loop(batcher=self.make_batcher(uids=uids), party=None)
        self.finalize()

    def report_metrics(self, y: DataTensor, predictions: DataTensor, name: str, step: int):
        y = y.numpy()
        predictions = predictions.detach().numpy()

        mae = metrics.mean_absolute_error(y, predictions)
        acc = ComputeAccuracy_numpy(is_linreg=False).compute(y, predictions)
        print(f"Master %s: %s metrics (MAE): {mae}" % (self.id, name))
        print(f"Master %s: %s metrics (Accuracy): {acc}" % (self.id, name))
        for avg in ["macro", "micro"]:
            try:
                roc_auc = roc_auc_score(y, predictions, average=avg)
            except ValueError:
                roc_auc = 0
            print(f'{name} ROC AUC {avg}: {roc_auc}')
            if self.run_mlflow:
                mlflow.log_metric(f"{name.lower()}_roc_auc_{avg}", roc_auc, step=step)

    def loop(self, batcher: Batcher, party) -> None:
        """ Perform training iterations using the given batcher.

        :param batcher: An iterable batch generator used for training.
        :return: None
        """
        for titer in batcher:
            predictions_delta = self.predict_partial(uids=titer.batch)  # d
            master_gradient = self.compute_gradient(predictions_delta, titer.batch)  # g_enc

            self.calculate_updates({'master': master_gradient})
            # self.update_weights(upd=update['master'], uids=titer.batch)
            master_predictions, targets = self.predict(uids=None)
            self.report_metrics(targets, master_predictions, 'Train', step=titer.seq_num)
            master_predictions, targets = self.predict(uids=None, is_test=True)
            self.report_metrics(targets, master_predictions, 'Test', step=titer.seq_num)

    def predict_partial(self, uids: List[str]):
        Xw, y = self.predict(uids, is_test=False)
        d = 0.25 * Xw - 0.5 * y
        return d

    def compute_gradient(self, aggregated_predictions_diff: DataTensor, uids: List[str]) -> DataTensor:
        X = self._dataset[self._data_params.train_split][self._data_params.features_key][[int(x) for x in uids]]
        g = 1 / X.shape[0] * torch.matmul(X.T, aggregated_predictions_diff)
        # g = g + self.alpha * self._model_parameter.weight.data.T

        return g

    # def update_weights(self, uids: List[str], upd: DataTensor):
    #     X = self._dataset[self._data_params.train_split][self._data_params.features_key][[int(x) for x in uids]]
    #     self.update_weights(X, upd, collected_from_arbiter=True)

    def calculate_updates(self, gradients: dict) -> dict[str, DataTensor]:
        master_gradient = gradients['master']
        size_list = [master_gradient.size()[0]]
        gradient = master_gradient.squeeze()
        self._optimizer_step(gradient.unsqueeze(1))
        # delta_gradients = self._get_delta_gradients()
        # splitted_grads = torch.tensor_split(delta_gradients, torch.cumsum(torch.tensor(size_list), 0)[:-1], dim=0)
        # deltas = {agent: splitted_grads[i] for i, agent in enumerate(['master'])}
        # return deltas

    def predict(self, uids: Optional[List[str]], is_test: bool = False):
        if not is_test:
            if uids is None:
                uids = self._uids_to_use
            X = self._dataset[self._data_params.train_split][self._data_params.features_key][[int(x) for x in uids]]
            y = self.target[[int(x) for x in uids]]
        else:
            X = self._dataset[self._data_params.test_split][self._data_params.features_key]
            y = self.test_target

        Xw = torch.matmul(X, self._model_parameter.weight.data.detach().T)
        print('Xw.shape', Xw.shape)

        return Xw, y

    def register_uids(self, uids: List[str]):
        self._uids_to_use = uids
