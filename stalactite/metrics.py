import torch
import numpy as np


class ComputeAccuracy:
    def __init__(self, name='Accuracy', positive_int=1, negative_int=-1, force_to_binary=True):
        """_summary_

        Args:
            name (str, optional): _description_. Defaults to 'Accuracy'.
            positive_int (int, optional): Depends on the ture labels. Defaults to 1.
            negative_int (int, optional): Depends on the ture labels. Defaults to -1.
            force_to_binary (bool, optional): _description_. Defaults to True.
        """
        self.force_to_binary = force_to_binary
        self.name = name
        self.positive_int = positive_int
        self.negative_int = negative_int

    def compute(self, true_label, predictions):
        predictions = predictions.detach().clone().squeeze()
        true_label = true_label.detach().clone().squeeze()

        if self.force_to_binary:
            predictions[predictions < 0] = self.negative_int
            predictions[predictions > 0] = self.positive_int
        # import pdb; pdb.set_trace()
        # accuracy = accuracy_score(true_label.numpy(), predictions.numpy())
        accuracy = torch.sum(true_label == predictions) / true_label.shape[0]
        return float(accuracy)


class ComputeAccuracy_numpy():
    def __init__(self, force_to_binary=True, is_linreg=True, is_multilabel=False):
        self.force_to_binary = force_to_binary
        self.name = 'Accuracy'
        self.is_linreg = is_linreg
        self.is_multilabel = is_multilabel

    @staticmethod
    def accuracy_multilable(y_true, y_pred):
        temp = 0
        for i in range(y_true.shape[0]):
            temp += sum(np.logical_and(y_true[i], y_pred[i])) / sum(np.logical_or(y_true[i], y_pred[i]))
        return temp / y_true.shape[0]

    def compute(self, true_label, predictions):
        predictions = predictions.copy()
        true_label = true_label.copy()

        if self.force_to_binary:
            if self.is_linreg:
                predictions[predictions < 0] = -1
                predictions[predictions > 0] = 1
            else:
                predictions[predictions < 0.5] = 0
                predictions[predictions > 0.5] = 1

        if self.is_multilabel:
            return self.accuracy_multilable(true_label, predictions)

        accuracy = np.sum(true_label.squeeze() == predictions.squeeze()) / true_label.squeeze().shape[0]
        return accuracy

