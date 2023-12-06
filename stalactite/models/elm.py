import numpy as np


def solve_ols_svd(U, S, Vh, Y, lamda=0.0):
    """
    Solve OLS problem given the SVD decomposition

    Input:
        ! Note X= U*S*Vh and Y are assumed to be normalized, hence lamda is between 0.0 and 1.0.

        U, S, Vh - SVD decomposition
        Y - target variables
        lamda - regularization parameter. Lamda must be normalized with respect
                                          to number of samples. Data is assumed
                                          to be normalized, so lamda is between 0.0 and 1.0.
    """

    n_points = U.shape[0]
    machine_epsilon = np.finfo(np.float64).eps
    if (lamda is None) or (lamda > 1.0) or (lamda < 0.0):  # no lamda optimization happened or wrong optim results
        num_rank = np.count_nonzero(S > S[0] * machine_epsilon)  # numerical rank

        # S.shape = (S.shape[0], 1) #todo: why this happens?
        coeffs = np.dot(Vh.T, np.multiply(1.0 / S, np.dot(U.T, Y)))

    else:

        S2 = np.power(S, 2)
        S2 = S2 + n_points * lamda  # parameter lambda normalized with respect to number of points !!!
        S = S / S2
        # S.shape = (S.shape[0], 1) #todo: why this happens?
        coeffs = np.dot(Vh.T, np.multiply(S, np.dot(U.T, Y)))

        num_rank = None  # numerical rank is None because regularization is used

    return coeffs, num_rank