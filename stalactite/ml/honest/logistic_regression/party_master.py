import logging
from typing import List

import mlflow
import torch
from sklearn.metrics import roc_auc_score, root_mean_squared_error

from stalactite.base import DataTensor, PartyDataTensor
from stalactite.ml.honest.linear_regression.party_master import HonestPartyMasterLinReg

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
sh = logging.StreamHandler()
sh.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
logger.addHandler(sh)


class HonestPartyMasterLogReg(HonestPartyMasterLinReg):
    """ Implementation class of the VFL honest PartyMaster specific to the Logistic Regression algorithm. """

    def make_init_updates(self, world_size: int) -> PartyDataTensor:
        """ Make initial updates for logistic regression.

        :param world_size: Number of party members.
        :return: Initial updates as a list of zero tensors.
        """
        logger.info("Master %s: making init updates for %s members" % (self.id, world_size))
        self._check_if_ready()
        # return [torch.zeros(self._batch_size, self.target.shape[1]) for _ in range(world_size)]
        return [torch.zeros(self._batch_size) for _ in range(world_size)]

    def aggregate(
            self, participating_members: List[str], party_predictions: PartyDataTensor, infer=False
    ) -> DataTensor:
        """ Aggregate party predictions for logistic regression.

        :param participating_members: List of participating party member identifiers.
        :param party_predictions: List of party predictions.
        :param infer: Flag indicating whether to perform inference.

        :return: Aggregated predictions after applying sigmoid function.
        """
        logger.info("Master %s: aggregating party predictions (num predictions %s)" % (self.id, len(party_predictions)))
        self._check_if_ready()
        if not infer:
            for member_id, member_prediction in zip(participating_members, party_predictions):
                self.party_predictions[member_id] = member_prediction
            party_predictions = list(self.party_predictions.values())
            predictions = torch.sum(torch.stack(party_predictions, dim=1), dim=1)
        else:
            predictions = self.activation(torch.sum(torch.stack(party_predictions, dim=1), dim=1))
        return predictions

    def compute_updates(
            self,
            participating_members: List[str],
            predictions: DataTensor,
            party_predictions: PartyDataTensor,
            world_size: int,
            uids: list[str],
    ) -> List[DataTensor]:
        """ Compute updates for logistic regression.

        :param participating_members: List of participating party members identifiers.
        :param predictions: Model predictions.
        :param party_predictions: List of party predictions.
        :param world_size: Number of party members.
        :param subiter_seq_num: Sub-iteration sequence number.

        :return: List of gradients as tensors.
        """
        logger.info("Master %s: computing updates (world size %s)" % (self.id, world_size))
        self._check_if_ready()
        self.iteration_counter += 1
        tensor_idx = [self._uid2tensor_idx[uid] for uid in uids]
        y = self.target[tensor_idx]
        # criterion = torch.nn.BCEWithLogitsLoss(pos_weight=self.class_weights) if len(y.shape) == 2 else torch.nn.CrossEntropyLoss(weight=self.class_weights)
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=self.class_weights) if self.binary else torch.nn.CrossEntropyLoss(weight=self.class_weights)
        targets_type = torch.LongTensor if isinstance(criterion,
                                                      torch.nn.CrossEntropyLoss) else torch.FloatTensor
        loss = criterion(torch.squeeze(predictions), y.type(targets_type))
        grads = torch.autograd.grad(outputs=loss, inputs=predictions)

        for i, member_id in enumerate(participating_members):
            self.updates[member_id] = grads[0]

        return [self.updates[member_id] for member_id in participating_members]

    def report_metrics(self, y: DataTensor, predictions: DataTensor, name: str, step: int) -> None:
        """Report metrics for logistic regression.

        Compute main classification metrics, if `use_mlflow` parameter was set to true, log them to MlFLow, log them to
        stdout.

        :param y: Target values.
        :param predictions: Model predictions.
        :param name: Name of the dataset ("Train" or "Test").

        :return: None.
        """
        logger.info(
            f"Master %s: reporting metrics. Y dim: {y.size()}. " f"Predictions size: {predictions.size()}" % self.id
        )

        y = y.numpy()
        predictions = predictions.detach().numpy()

        if self.binary:
            for avg in ["macro", "micro"]:
                try:
                    roc_auc = roc_auc_score(y, predictions, average=avg)
                except ValueError:
                    roc_auc = 0

                rmse = root_mean_squared_error(y, predictions)

                logger.info(f'{name} RMSE on step {step}: {rmse}')
                logger.info(f'{name} ROC AUC {avg} on step {step}: {roc_auc}')
                if self.run_mlflow:
                    mlflow.log_metric(f"{name.lower()}_roc_auc_{avg}", roc_auc, step=step)
                    mlflow.log_metric(f"{name.lower()}_rmse", rmse, step=step)
        else:
            avg = "macro"
            try:
                roc_auc = roc_auc_score(y, predictions, average=avg, multi_class="ovr")
            except ValueError:
                roc_auc = 0

            logger.info(f'{name} ROC AUC {avg} on step {step}: {roc_auc}')

            if self.run_mlflow:
                mlflow.log_metric(f"{name.lower()}_roc_auc_{avg}", roc_auc, step=step)
