import numpy as np
from numba import jit
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from numpy.linalg import norm

from ad.queue import Queue


class SingularSpectrumTransformAD:

    def __init__(self, win_length: int, n_components: int = 3, order: int = 100, lag: int = 0,
                 scaling_used: bool = True, n_search_step: int = 1, use_lanczos=True, rank_lanczos=None, eps=1e-3):
        """
        Initialization about the change point detection with Singular Spectrum Transformation
        :param win_length: window length of Hankel matrix
        :param n_components: the rank of Hankel matrix will be taken
        :param order: number of columns of Hankel matrix
        :param lag: lag between the history and test series
        :param scaling_used: if use min-max scaler to scale the input x
        :param n_search_step: the step of the change-point searching
        :param use_lanczos: if use lanczos algorithm
        :param rank_lanczos:  the rank which will be used for lanczos method.
        :param eps:  the eps which will be used for lanczos method.
        """
        self.win_length = win_length
        self.scaling_used = scaling_used
        self.n_components = n_components
        self.order = order
        self.lag = lag

        self.n_search_step = n_search_step

        self.use_lanczos = use_lanczos
        self.rank_lanczos = rank_lanczos
        self.eps = eps

        self._data_cached_queue = None
        self._scores_cached_queue = None

        # store history_data

    def fit_transform(self, x: np.array):
        """
        Calculate the x's anomaly score.(offline)
        :param x: 1d numpy array
        :return: 1d array change point score
        """
        # normal rule for params
        if self.order is None:
            # the same of before and after series, and it's better to set the best number by users
            self.order = self.win_length
        if self.lag is None:
            # best self.order // 2, funnel is self.order + self.win_length + 1
            self.lag = self.order // 2
        if self.rank_lanczos is None:
            if self.n_components % 2 == 0:
                self.rank_lanczos = 2 * self.n_components
            else:
                self.rank_lanczos = 2 * self.n_components - 1

        # validate all the params
        assert x.ndim == 1, "input x dimension must be 1"
        assert self.win_length + self.order + self.lag < x.size, "data length is too short."

        if self.scaling_used:
            x_scaled = MinMaxScaler().fit_transform(x.reshape(-1, 1))[:, 0]
        else:
            x_scaled = x

        # call the function for getting the change point scores
        scores = _calc_sst_score(x_scaled, self.win_length, self.n_components, self.order, self.lag, self.n_search_step,
                                 self.use_lanczos, self.rank_lanczos, self.eps
                                 )

        return scores

    def fit_predict(self, x: np.array, is_only_detect_last_one: bool = False) -> (np.array, np.array):

        if is_only_detect_last_one:
            x = x[-(self.win_length + self.order + self.lag + 1):]
        scores = self.fit_transform(x)
        scores = _robust_sst_scores(x, scores, self.win_length, self.order, self.lag)

        if is_only_detect_last_one:
            if self._scores_cached_queue is None:
                self._scores_cached_queue = Queue()
            score = scores[-1]
            if score != 0:
                self._scores_cached_queue.enqueue(score)
            if self._scores_cached_queue.size > 10:
                scores_non_zero = np.array(self._scores_cached_queue.get_all_data())
                nu = np.mean(scores_non_zero)
                sigma = np.std(scores_non_zero)
                scores_non_zero = np.abs(scores_non_zero)
                new_anomalies = np.where(scores_non_zero - nu - 3 * sigma > 0, -1, 1)
                return [new_anomalies[-1]], [score]
            else:
                return [0], [score]
        else:
            scores_non_zero = scores[self.win_length + self.order + self.lag:]
            anomalies = np.zeros(self.win_length + self.order + self.lag)
            nu = np.mean(scores_non_zero)
            sigma = np.std(scores_non_zero)
            scores_non_zero = np.abs(scores_non_zero)
            new_anomalies = np.where(scores_non_zero - nu - 3 * sigma > 0, -1, 1)
            anomalies = np.append(anomalies, new_anomalies)
            return anomalies, scores

    def fit_predict_online(self, x: float):
        if self._data_cached_queue is None:
            self._data_cached_queue = Queue(queue_size=2000)
        self._data_cached_queue.enqueue(x)

        detect_x_series = np.array(self._data_cached_queue.get_all_data())

        if detect_x_series.size < self.win_length + self.order + self.lag + 1:
            return 0, 0

        anomalies, scores = self.fit_predict(detect_x_series, is_only_detect_last_one=True)
        return anomalies[-1], scores[-1]


@jit(nopython=True)
def _robust_sst_scores(x, scores, win_length, order, lag):
    for i in range(x.size - (win_length + order + lag)):
        median_t_a = np.median(x[i:i + order + win_length + 1])
        median_t_b = np.median(x[i + lag:i + order + win_length + lag + 1])
        median_adj_factor = np.abs(median_t_b - median_t_a)
        median_diff_a = np.sqrt(np.median(np.abs(x[i:i + order + win_length + 1] - median_t_a)))
        median_diff_b = np.sqrt(
            np.median(np.abs(x[i + lag:i + order + win_length + lag + 1] - median_t_b)))
        median_diff_adj_factor = np.abs(median_diff_b - median_diff_a)
        scores[i + win_length + order + lag] *= median_diff_adj_factor * median_adj_factor
    return scores


@jit(nopython=True)
def _construct_hankel_matrix(x, order, start, end):
    """
    construct hankel matrix
    :param x: the series
    :param order: number of columns of Hankel matrix
    :param start: the start index of the data
    :param end: the end index of the data
    :return:
    """
    l, k = end - start, order
    matrix_x = np.empty((l, k))
    for i in range(order):
        matrix_x[:, i] = x[(start - i):(end - i)]
    return matrix_x


@jit("f8(f8[:,:],f8[:,:],u1)", nopython=True)
def _sst_svd(h_matrix_after, h_matrix_before, n_components):
    """
    compute sst scores by svd algorithm
    :param h_matrix_after: the h-matrix before t
    :param h_matrix_before: the h-matrix after t
    :param n_components: the number of feature for calculation
    :return: the scores for the difference of the two series
    """
    # svd
    u_after, _, _ = np.linalg.svd(h_matrix_after, full_matrices=False)
    u_before, _, _ = np.linalg.svd(h_matrix_before, full_matrices=False)
    # calculate the eig-value of the corr of the two matrix
    _, s, _ = np.linalg.svd(u_after[:, :n_components].T @
                            u_before[:, :n_components], full_matrices=False)
    # use the first eigen value
    return 1 - s[0]


@jit(nopython=True)
def _calc_sst_score(x, win_length, n_components, order, lag, n_search_step, use_lanczos, rank, eps):
    """
    Compute the sst scores
    """
    start_idx = win_length + order + lag + 1
    end_idx = x.size + 1

    if use_lanczos:
        # x0 = np.empty(order, dtype=np.float64)
        x0 = np.random.rand(order)
        x0 /= np.linalg.norm(x0)

    score = np.zeros_like(x)
    # for the data (> the min index which can be computed)
    for t in range(start_idx, end_idx):
        # skip the search step
        if t % n_search_step == 0:
            # construct the Hankel matrix A(after t+lag) and B(before t)
            h_matrix_before = _construct_hankel_matrix(x, order, start=t - win_length - lag, end=t - lag)
            h_matrix_after = _construct_hankel_matrix(x, order, start=t - win_length, end=t)
            # compute the anomaly scores
            if use_lanczos:
                # use lanczos algorithm
                score[t - 1], x1 = _sst_lanczos(h_matrix_after, h_matrix_before, n_components,
                                                rank, x0)
                # update initial vector for power method
                x0 = x1 + eps * np.random.rand(x0.size)
                x0 /= np.linalg.norm(x0)
            else:
                # use svd algorithm
                score[t - 1] = _sst_svd(h_matrix_after, h_matrix_before, n_components)
        elif t - 2 >= 0:
            score[t - 1] = score[t - 2]
        else:
            score[t - 1] = 0

    return score


@jit(nopython=True)
def _sst_lanczos(h_matrix_after, h_matrix_before, n_components, rank, x0):
    """
    Run sst algorithm with lanczos method (in the FELIX-SST algorithm and IKA(FUNNEL) algorithm).
    """
    P_before = h_matrix_before.T @ h_matrix_before
    P_after = h_matrix_after.T @ h_matrix_after
    # calculate the first singular vec of test matrix
    u, _, _ = power_method(P_after, x0, n_iter=1)
    T = lanczos(P_before, u, rank)
    vec, val = eig_tridiag(T)
    return 1 - (vec[0, :n_components] ** 2).sum(), u


@jit(nopython=True)
def power_method(A, x0, n_iter=1):
    for i in range(n_iter):
        x0 = A.T @ A @ x0

    v = x0 / norm(x0)
    s = norm(A @ v)
    u = A @ v / s

    return u, s, v


@jit(nopython=True)
def lanczos(C, a, s):
    # initialization
    r = np.copy(a)
    a_pre = np.zeros_like(a, dtype=np.float64)
    beta_pre = 1
    T = np.zeros((s, s))

    for j in range(s):
        a_post = r / beta_pre
        alpha = a_post.T @ C @ a_post
        r = C @ a_post - alpha * a_post - beta_pre * a_pre
        beta_post = norm(r)

        T[j, j] = alpha
        if j - 1 >= 0:
            T[j, j - 1] = beta_pre
            T[j - 1, j] = beta_pre

        a_pre = a_post
        beta_pre = beta_post

    return T


@jit(nopython=True)
def eig_tridiag(T):
    """
    Compute eigen value decomposition for tridiag matrix.
    TODO: efficient implementation
    :param T: matrix
    :return: eigen vector and eigen value
    """
    u, s, _ = np.linalg.svd(T)
    return u, s


def plot_data_and_score(raw_data, score):
    f, ax = plt.subplots(2, 1, figsize=(20, 10))
    ax[0].plot(raw_data);
    ax[0].set_title("raw data")
    ax[1].plot(score, "r")
    ax[1].set_title("score")
    plt.show()


if __name__ == "__main__":
    # synthetic (frequency change)
    n = 1000
    x0 = np.sin(2 * np.pi * 1 * np.linspace(0, 10, n))
    x1 = np.sin(2 * np.pi * 2 * np.linspace(0, 10, n))
    x2 = np.sin(2 * np.pi * 8 * np.linspace(0, 10, n))
    x = np.hstack([x0, x1, x2])
    x += + np.random.rand(x.size)

    # import time
    #
    # start_time = time.time()
    # model = SingularSpectrumTransformAD(win_length=60, order=20, lag=80)
    # scores = model.fit_transform(x)
    # end_time = time.time()
    # print("eclapse: {}s".format(end_time - start_time))
    #
    # plot_data_and_score(x, scores)

    model = SingularSpectrumTransformAD(win_length=60, order=20, lag=5)
    anomalies, scores = model.fit_predict(x)
    plot_data_and_score(x, scores)

    for idx, xi in enumerate(x):
        a, s = model.fit_predict_online(xi)
        if a == -1:
            print(idx, xi, a, s)
