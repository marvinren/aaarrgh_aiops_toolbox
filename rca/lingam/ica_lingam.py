import pandas as pd
import numpy as np
from lingam.utils import predict_adaptive_lasso
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import FastICA
from sklearn.utils import check_array


class ICALiNGAM:
    """Implementation of ICA-based LiNGAM Algorithm, It's a base method for LiNGAM
    References
    ----------
    .. [1] S. Shimizu, P. O. Hoyer, A. Hyvärinen, and A. J. Kerminen.
       A linear non-gaussian acyclic model for causal discovery.
       Journal of Machine Learning Research, 7:2003-2030, 2006.
    """

    def __init__(self, max_iter=1000):
        """
        construct a ICA-based LiNGAM model
        :param max_iter: the maximum number of iter of FastICA
        """
        self._max_iter = max_iter
        self._causal_order = None
        self._adjacency_matrix = None

    def fit(self, X: pd.DataFrame):
        """
        Fit the model to X.
        :param X:
        :return:
        """
        X = check_array(X)

        # get the independent component by fastICA
        ica = FastICA(max_iter=self._max_iter, random_state=None)
        ica.fit(X)
        W = ica.components_

        # get a permuted W
        _, col_index = linear_sum_assignment(1 / np.abs(W))
        permutated_W = np.zeros_like(W)
        permutated_W[col_index] = W

        # get a vector to scale
        D = np.diag(permutated_W)[:, np.newaxis]

        # get a adjacency matrix B
        W_estimate = permutated_W / D
        B_estimate = np.eye(len(W_estimate)) - W_estimate

        print(B_estimate)

        # find a casual order
        causal_order = self._estimate_causal_order(B_estimate)
        self._causal_order = causal_order
        print(causal_order)

        return self._estimate_adjacency_matrix(X)

    def _estimate_causal_order(self, matrix):
        """Obtain a lower triangular from the given matrix approximately.
        Parameters
        ----------
        matrix : array-like, shape (n_features, n_samples)
            Target matrix.
        Return
        ------
        causal_order : array, shape [n_features, ]
            A causal order of the given matrix on success, None otherwise.
        """
        causal_order = None

        # set the m(m + 1)/2 smallest(in absolute value) elements of the matrix to zero
        pos_list = np.argsort(np.abs(matrix), axis=None)
        pos_list = np.vstack(np.unravel_index(pos_list, matrix.shape)).T
        initial_zero_num = int(matrix.shape[0] * (matrix.shape[0] + 1) / 2)
        for i, j in pos_list[:initial_zero_num]:
            matrix[i, j] = 0

        for i, j in pos_list[initial_zero_num:]:
            # set the smallest(in absolute value) element to zero
            matrix[i, j] = 0

            causal_order = self._search_causal_order(matrix)
            if causal_order is not None:
                break

        return causal_order

    def _search_causal_order(self, matrix):
        """Obtain a causal order from the given matrix strictly.
        Parameters
        ----------
        matrix : array-like, shape (n_features, n_samples)
            Target matrix.
        Return
        ------
        causal_order : array, shape [n_features, ]
            A causal order of the given matrix on success, None otherwise.
        """
        causal_order = []

        row_num = matrix.shape[0]
        original_index = np.arange(row_num)


        while 0 < len(matrix):
            # find a row all of which elements are zero
            row_index_list = np.where(np.sum(np.abs(matrix), axis=1) == 0)[0]
            if len(row_index_list) == 0:
                break

            target_index = row_index_list[0]

            # append i to the end of the list
            causal_order.append(original_index[target_index])
            original_index = np.delete(original_index, target_index, axis=0)

            # remove the i-th row and the i-th column from matrix
            mask = np.delete(np.arange(len(matrix)), target_index, axis=0)
            matrix = matrix[mask][:, mask]

        if len(causal_order) != row_num:
            causal_order = None

        return causal_order

    def _estimate_adjacency_matrix(self, X, prior_knowledge=None):
        """Estimate adjacency matrix by causal order.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.
        prior_knowledge : array-like, shape (n_variables, n_variables), optional (default=None)
            Prior knowledge matrix.
        Returns
        -------
        self : object
            Returns the instance itself.
        """
        if prior_knowledge is not None:
            pk = prior_knowledge.copy()
            np.fill_diagonal(pk, 0)

        B = np.zeros([X.shape[1], X.shape[1]], dtype="float64")
        for i in range(1, len(self._causal_order)):
            target = self._causal_order[i]
            predictors = self._causal_order[:i]

            # Exclude variables specified in no_path with prior knowledge
            if prior_knowledge is not None:
                predictors = [p for p in predictors if pk[target, p] != 0]

            # target is exogenous variables if predictors are empty
            if len(predictors) == 0:
                continue

            B[target, predictors] = predict_adaptive_lasso(X, predictors, target)

        self._adjacency_matrix = B
        return self


if __name__ == "__main__":
    x3 = np.random.uniform(size=1000)
    x0 = 3.0 * x3 + np.random.uniform(size=1000)
    x2 = 6.0 * x3 + np.random.uniform(size=1000)
    x1 = 3.0 * x0 + 2.0 * x2 + np.random.uniform(size=1000)
    x5 = 4.0 * x0 + np.random.uniform(size=1000)
    x4 = 8.0 * x0 - 1.0 * x2 + np.random.uniform(size=1000)
    X = pd.DataFrame(np.array([x0, x1, x2, x3, x4, x5]).T, columns=['x0', 'x1', 'x2', 'x3', 'x4', 'x5'])
    print("the origin data is: \n")
    print(X.head())

    model = ICALiNGAM()
    model.fit(X)
    print(model._adjacency_matrix)