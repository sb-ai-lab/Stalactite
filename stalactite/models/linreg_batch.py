import logging

import scipy as sp
import torch

from .elm import solve_ols_svd

logger = logging.getLogger(__name__)


class LinearRegressionBatch(torch.nn.Module):
    def __init__(self, input_dim, output_dim, reg_lambda=0.0, **kw):
        """

        Args:
            input_dim (int): input dimension
            output_dim (int): output dimension
            reg_lambda (float): L2 regularizer per 1-sample. That is - regularizer on the scale of
                MSE error.
        """
        super(LinearRegressionBatch, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim, bias=False, device=None, dtype=None)
        self.reg_lambda = reg_lambda
        self.requires_grad_(False)  # turn off gradient computation
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.reg_lambda = reg_lambda

    def forward(self, x):
        # try:
        outputs = self.linear(x)
        # except:
        #     print(x.size())
        #     import pdb
        #     pdb.set_trace()
        return outputs

    def update_weights(self, X_train, rhs, optimizer=None) -> None:
        # todo: add docs
        U, S, Vh = sp.linalg.svd(X_train.cpu().numpy(), full_matrices=False, overwrite_a=False, check_finite=False)
        logger.debug("Updating weights inside model")
        coeffs, num_rank = solve_ols_svd(U, S, Vh, rhs.cpu(), self.reg_lambda)
        self.linear.weight.copy_(torch.as_tensor(coeffs).t())  # TODO: copying is not efficient
        logger.debug("Success: update weights")

    def predict(self, X_pred):
        Y_pred = self.forward(X_pred)
        return Y_pred

    def get_weights(
        self,
    ):
        # import pdb; pdb.set_trace()
        weights = self.linear.weight.clone()
        return weights

    @property
    def init_params(self):
        return {
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'reg_lambda': self.reg_lambda,
        }
