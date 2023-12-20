import logging

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

    def forward(self, x):
        # try:
        outputs = self.linear(x)
        # except:
        #     print(x.size())
        #     import pdb
        #     pdb.set_trace()
        return outputs

    def update_weights(self, data_U, data_S, data_Vh, rhs):
        """Update weights from the SVD of data.

        Args:
            data_U (tensor): U of SVD of data matrix.
            data_S (tensor): S of SVD of data matrix.
            data_Vh (tensor): Vh of SVD of data matrix.
            rhs (tensor): Y minus prediction of other parties

        Returns:
            coeffs (tensor): new vector of coefficients
        """
        logger.debug("updating weights inside model")
        coeffs, num_rank = solve_ols_svd(data_U, data_S, data_Vh, rhs, self.reg_lambda)
        self.linear.weight.copy_(torch.as_tensor(coeffs).t())  # TODO: copying is not efficient
        logger.debug("SUCCESS update weights")
        return coeffs

    def predict(self, X_pred):
        Y_pred = self.forward(X_pred)
        return Y_pred

    def get_weights(self, ):
        # import pdb; pdb.set_trace()
        weights = self.linear.weight.clone()
        return weights
