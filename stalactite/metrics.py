import torch


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
