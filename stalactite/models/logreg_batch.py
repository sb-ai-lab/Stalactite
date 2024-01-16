import logging

import torch

from .elm import solve_ols_svd

logger = logging.getLogger(__name__)


class LogisticRegressionBatch(torch.nn.Module):
    def __init__(self, input_dim, output_dim, learning_rate=0.001, **kw):
        """
        Args:
            input_dim (int): input dimension
            output_dim (int): output dimension
        """
        super(LogisticRegressionBatch, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim, bias=False, device=None, dtype=None)
        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.SGD(self.linear.parameters(), lr=learning_rate)
        # self.optimizer = torch.optim.Adam(self.linear.parameters(), lr=learning_rate)


    def forward(self, x):
        return self.linear(x)

    def update_weights(self, X, grads, is_single=False) -> None:
        # todo: add docs
        logger.debug("updating weights inside model")
        self.optimizer.zero_grad()
        logits = self.forward(X)
        if is_single:
            loss = self.criterion(torch.squeeze(logits), grads.float())
            loss.backward()
        else:
            logits.backward(gradient=grads)
        self.optimizer.step()
        logger.debug("SUCCESS update weights")

    def predict(self, X_pred):
        Y_pred = self.forward(X_pred)
        return Y_pred

    def get_weights(self, ):
        weights = self.linear.weight.clone()
        return weights
