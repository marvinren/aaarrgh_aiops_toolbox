import math

import numpy as np
from scipy.optimize import minimize
from numba import jit

from ad.datagen import AnomalyUnivariateGen


class SPOTAD:
    """Univariate Spot Anomaly Detector"""

    def __init__(self, init_length: int = 300, window_size: int = 10):
        self.init_length = init_length
        self.window_size = window_size
        self.prob = 1e-4

        # initialization of the history concept params
        # TODO: the history_data(init_cache_data) is not stored all, but the peaks data is all saved.

        # n (number of the observed x)
        self.n_x = 0
        # the observed x queue.
        self.init_cache_data = []
        # t (threshold)
        self.threshold_up = None
        self.threshold_down = None
        # peak-over values
        self.peaks_up = None
        self.peaks_down = None
        # Nt (number of peak-over values)
        self.n_threshold_up = None
        self.n_threshold_down = None
        # sigma
        self.sigma_up = None
        self.sigma_down = None
        # gamma
        self.gamma_up = None
        self.gamma_down = None
        # anomaly threshold(zq)
        self.zq_up = None
        self.zq_down = None

        self.thup = []
        self.thdown = []

    def fit_predict(self, x: np.ndarray):
        # TODO: not completed
        assert x.ndim == 1, "the detect data's dim is not equal to 1"
        return None, None

    def fit_predict_online(self, v: float):
        # add observed x , n + 1 and add value to cache queue
        self.n_x += 1
        self.init_cache_data.append(v)

        # if the size of observed x is greater than the init_x_size, compute the drift values
        if self.n_x == self.init_length:
            # calculate the threshold(t) and peaks after the concept drift
            self._init_drift(np.array(self.init_cache_data), self.window_size)
            # Initialize: calculate the zq, sigma, gamma
            self.gamma_up, self.sigma_up, _ = _grimshaw(self.peaks_up)
            self.gamma_down, self.sigma_down, _ = _grimshaw(self.peaks_down)
            self.zq_up = self._calc_evt_zq_threshold_up(self.sigma_up, self.gamma_up)
            self.zq_down = self._calc_evt_zq_threshold_down(self.sigma_down, self.gamma_down)

        if self.n_x >= self.init_length:
            hist_mean = np.mean(self.init_cache_data[-self.window_size:])
            normal_X = v - hist_mean
            prob = 0.0
            if normal_X > self.zq_up:
                prob = 1.0
                self.init_cache_data = self.init_cache_data[:-1]
            elif normal_X > self.threshold_up:
                prob = float(normal_X - self.threshold_up) / (self.zq_up - self.threshold_up)
                self.peaks_up = np.append(self.peaks_up, normal_X - self.threshold_up)
                self.n_threshold_up += 1
                self.gamma_up, self.sigma_up, _ = _grimshaw(self.peaks_up)
                self.zq_up = self._calc_evt_zq_threshold_up(self.sigma_up, self.gamma_up)
            elif normal_X < self.zq_down:
                prob = 1.0
                self.init_cache_data = self.init_cache_data[:-1]
            elif normal_X < self.threshold_down:
                prob = (self.threshold_down - normal_X) / (self.zq_down - normal_X)
                self.peaks_down = np.append(self.peaks_down, self.threshold_down - normal_X)
                self.n_threshold_down += 1

                self.gamma_down, self.sigma_down, _ = _grimshaw(self.peaks_down)
                self.zq_down = self._calc_evt_zq_threshold_down(self.sigma_down, self.gamma_down)
            else:
                prob = 0.0

            self.init_cache_data = self.init_cache_data[-self.window_size:]

            self.thup.append(self.zq_up + hist_mean)
            self.thdown.append(self.zq_down + hist_mean)
            if prob == 0.0 :
                return 1, prob
            else:
                return -1, prob

        return 0, 0

    def _calc_evt_zq_threshold_up(self, sigma_up, gamma_up):
        r_up = (
                (self.init_length - self.window_size)
                * self.prob
                / self.n_threshold_up
        )
        if gamma_up != 0:
            zq_up = self.threshold_up * (sigma_up / gamma_up) * (math.pow(r_up, -gamma_up) - 1)
        else:
            zq_up = self.threshold_up - sigma_up * math.log(r_up)

        return zq_up

    def _calc_evt_zq_threshold_down(self, sigma_down, gamma_down):
        r_down = (
                (self.init_length - self.window_size)
                * self.prob
                / self.n_threshold_down
        )
        if gamma_down != 0:
            zq_down = self.threshold_down - (sigma_down / gamma_down) * (pow(r_down, -gamma_down) - 1)
        else:
            zq_down = self.threshold_down + sigma_down * math.log(r_down)
        return zq_down

    def _init_drift(self, history_data, window_size):
        threshold_up = None
        threshold_down = None
        peaks_up = None
        peaks_down = None

        # calc the new M value
        M = []
        w = sum(history_data[:window_size])
        M.append(w / window_size)
        for i in range(window_size, history_data.size):
            w = w - history_data[i - window_size] + history_data[i]
            M.append(w / window_size)
        # cals the threshold(t) values(up and down)
        thresholds = history_data[window_size:] - M[:-1]
        # sorted_thresholds = np.sort(thresholds.tolist())
        # threshold_up = sorted_thresholds[int(0.98*sorted_thresholds.size)]
        # threshold_down = sorted_thresholds[int(0.02 * sorted_thresholds.size)]
        self.threshold_up = np.quantile(thresholds, 0.98)
        self.threshold_down = np.quantile(thresholds, 0.02)
        # get the up/down peak values
        self.peaks_up = thresholds[thresholds > self.threshold_up] - self.threshold_up
        self.peaks_down = self.threshold_down - thresholds[thresholds < self.threshold_down]

        self.n_threshold_up = self.peaks_up.size
        self.n_threshold_down = self.peaks_down.size

        return self


def _grimshaw(peaks, epsilon=1e-8, n_points=8):
    def u(s):
        return 1 + np.log(s).mean()

    def v(s):
        return np.mean(1 / s)

    def w(Y, t):
        s = 1 + t * Y
        us = u(s)
        vs = v(s)
        return us * vs - 1

    def jac_w(Y, t):
        s = 1 + t * Y
        us = u(s)
        vs = v(s)
        jac_us = (1 / t) * (1 - vs)
        jac_vs = (1 / t) * (-vs + np.mean(1 / s ** 2))
        return us * jac_vs + vs * jac_us

    Ym = peaks.min()
    YM = peaks.max()
    Ymean = peaks.mean()

    a = -1 / YM
    if abs(a) < 2 * epsilon:
        epsilon = abs(a) / n_points

    a = a + epsilon
    b = 2 * (Ymean - Ym) / (Ymean * Ym)
    c = 2 * (Ymean - Ym) / (Ym ** 2)

    left_zeros = _rootsFinder(
        lambda t: w(peaks, t),
        lambda t: jac_w(peaks, t),
        (a + epsilon, -epsilon),
        n_points,
        "regular",
    )

    right_zeros = _rootsFinder(
        lambda t: w(peaks, t),
        lambda t: jac_w(peaks, t),
        (b, c),
        n_points,
        "regular",
    )

    # all the possible roots
    zeros = np.concatenate((left_zeros, right_zeros))

    # 0 is always a solution so we initialize with it
    gamma_best = 0
    sigma_best = Ymean
    ll_best = _log_likelihood(peaks, gamma_best, sigma_best)

    # we look for better candidates
    for z in zeros:
        gamma = u(1 + z * peaks) - 1
        sigma = gamma / z
        ll = _log_likelihood(peaks, gamma, sigma)
        if ll > ll_best:
            gamma_best = gamma
            sigma_best = sigma
            ll_best = ll
    return gamma_best, sigma_best, ll_best


def _rootsFinder(fun, jac, bounds, npoints, method):
    if method == "regular":
        step = (bounds[1] - bounds[0]) / (npoints + 1)
        try:
            X0 = np.arange(bounds[0] + step, bounds[1], step)
        except:
            X0 = np.random.uniform(bounds[0], bounds[1], npoints)
    elif method == "random":
        X0 = np.random.uniform(bounds[0], bounds[1], npoints)

    def objFun(X, f, jac):
        g = 0
        j = np.zeros(X.shape)
        i = 0
        for x in X:
            fx = f(x)
            g = g + fx ** 2
            j[i] = 2 * fx * jac(x)
            i = i + 1
        return g, j

    opt = minimize(
        lambda X: objFun(X, fun, jac),
        X0,
        method="L-BFGS-B",
        jac=True,
        bounds=[bounds] * len(X0),
    )

    X = opt.x
    np.round(X, decimals=5)
    return np.unique(X)


@jit(nopython=True)
def _log_likelihood(Y, gamma, sigma):
    n = Y.size
    if gamma != 0:
        tau = gamma / sigma
        L = -n * math.log(sigma) - (1 + (1 / gamma)) * (np.log(1 + tau * Y)).sum()
    else:
        L = n * (1 + math.log(Y.mean()))
    return L


if __name__ == '__main__':

    gen = AnomalyUnivariateGen(n=3000, anomaly_prob=0.01)
    data, labels = gen.generate()
    model = SPOTAD()
    for i, x in enumerate(data):
        l, p = model.fit_predict_online(x)
        if labels[i] == -1:
            print(-1, l, p)
        elif l==-1:
            print(l, p)
