import numpy as np


class CUSUMChangeDetection:
    """
    通过CUSUM算法，对单指标时序数据进行变点检测
    Detection of changes using the Cumulative Sum (CUSUM)
    reference: https://nbviewer.org/github/BMClab/BMC/blob/master/notebooks/DetectCUSUM.ipynb
    """

    def __init__(self, threshold=1, drift=0, ending=False):
        # base configuration
        self.threshold = threshold
        self.drift = drift
        self.ending = ending

        # x values
        self._x = None
        # g+ and g- function values
        self._gp = None
        self._gn = None
        # change point index (alarm point)
        self._ta = None
        # change start index
        self._tai = None
        # change end index
        self._taf = None
        # amplitude of change time period
        self._amp = None

    def fit(self, x):
        return self

    def predict(self, x):
        x = np.atleast_1d(x).astype('float64')
        self._x = x

        self._amp = np.array([])
        self._gp, self._gn, self._ta, self._tai, self._taf = self._cusum_detect(x)

        if self._tai.size and self.ending:
            _, _, tai2, _, _ = self._cusum_detect(x[::-1])
            self._taf = x.size - tai2[::-1] - 1
            self._tai, ind = np.unique(self._tai, return_index=True)
            self._ta = self._ta[ind]

            #
            if self._tai.size != self._taf.size:
                if self._tai.size < self._taf.size:
                    self._taf = self._taf[[np.argmax(self._taf >= i) for i in self._ta]]
                else:
                    ind = [np.argmax(i >= self._ta[::-1]) - 1 for i in self._taf]
                    self._ta = self._ta[ind]
                    self._tai = self._tai[ind]

            # Delete intercalated changes (the ending of the change is after
            # the beginning of the next change)
            ind = self._taf[:-1] - self._tai[1:] > 0
            if ind.any():
                self._ta = self._ta[~np.append(False, ind)]
                self._tai = self._tai[~np.append(False, ind)]
                self._taf = self._taf[~np.append(ind, False)]
            # Amplitude of changes
            self._amp = x[self._taf] - x[self._tai]

        return self._ta, self._tai, self._taf, self._amp

    def fit_predict(self, x):
        return self.fit(x).predict(x)

    def _cusum_detect(self, x):
        x = np.atleast_1d(x).astype('float64')

        gp, gn = np.zeros(x.size), np.zeros(x.size)
        ta, tai, taf = np.array([[], [], []], dtype=int)
        tap, tan = 0, 0

        for i in range(1, x.size):
            # calc s value
            s = x[i] - x[i - 1]
            # calc g+(g positive) value
            gp[i] = gp[i - 1] + s - self.drift
            # calc g-(g negative) value
            gn[i] = gp[i - 1] - s - self.drift
            # equals to "gp[i] = max(gp[i], 0)"
            if gp[i] < 0:
                gp[i], tap = 0, i
            # equals to "gn[i] = max(gn[i], 0)"
            if gn[i] < 0:
                gn[i], tan = 0, i
            # if gp or gn > threshold, add alarm by index and reset gp and gn
            if gp[i] > self.threshold or gn[i] > self.threshold:
                #print(gp[i], gn[i], self.threshold, tap, tan)
                ta = np.append(ta, i)  # alarm index
                tai = np.append(tai, tap if gp[i] > self.threshold else tan)
                # reset gp gn
                gp[i], gn[i] = 0, 0
        return gp, gn, ta, tai, taf

    def plot(self, plt=None):

        if plt is None:
            try:
                import matplotlib.pyplot as plt
                # plt.rcParams['axes.facecolor'] = 'white'
                plt.rcParams['figure.facecolor'] = 'white'
            except ImportError:
                print('matplotlib is not available.')
                return

        _, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))
        x = self._x
        t = range(x.size)
        ax1.plot(t, x, 'b-', lw=2)
        if len(self._ta):
            ax1.plot(self._tai, x[self._tai], '>', mfc='g', mec='g', ms=10,
                     label='Start')
            if self.ending:
                ax1.plot(self._taf, x[self._taf], '<', mfc='g', mec='g', ms=10,
                         label='Ending')
            ax1.plot(self._ta, x[self._ta], 'o', mfc='r', mec='r', mew=1, ms=5,
                     label='Alarm')
            ax1.legend(loc='best', framealpha=.5, numpoints=1)
        ax1.set_xlim(-.01 * x.size, x.size * 1.01 - 1)
        ax1.set_xlabel('Data #', fontsize=14)
        ax1.set_ylabel('Amplitude', fontsize=14)
        ymin, ymax = x[np.isfinite(x)].min(), x[np.isfinite(x)].max()
        yrange = ymax - ymin if ymax > ymin else 1
        ax1.set_ylim(ymin - 0.1 * yrange, ymax + 0.1 * yrange)
        ax1.set_title('Time series and detected changes ' +
                      '(threshold= %.3g, drift= %.3g): N changes = %d'
                      % (self.threshold, self.drift, len(self._tai)))
        ax2.plot(t, self._gp, 'y-', label='+')
        ax2.plot(t, self._gn, 'm-', label='-')
        ax2.set_xlim(-.01 * x.size, x.size * 1.01 - 1)
        ax2.set_xlabel('Data #', fontsize=14)
        ax2.set_ylim(-0.01 * self.threshold, 1.1 * self.threshold)
        ax2.axhline(self.threshold, color='r')
        ax1.set_ylabel('Amplitude', fontsize=14)
        ax2.set_title('Time series of the cumulative sums of ' +
                      'positive and negative changes')
        ax2.legend(loc='best', framealpha=.5, numpoints=1)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    detector = CUSUMChangeDetection(2, 3, True)

    x = np.random.randn(300)
    x[100:200] += 6
    print(np.var(x))
    ta, tai, taf, amp = detector.fit_predict(x)
    detector.plot()

