from typing import Union

from anomaly_detection.base import BaseDetector
import numpy as np
import pandas as pd

class ZScoreDetector(BaseDetector):
    def __init__(self, window_len: int = 100, is_global: bool = False):
        super().__init__()

        self.data_type = "univariate"
        self.window_len = window_len
        # self.stat = StreamStatistic(is_global=is_global, window_len=window_len)

    def fit(self, X: Union[np.ndarray, pd.DataFrame]):
        self.stat.update(X[0])
        return self

    def score(self, X: np.ndarray):
        mean = self.stat.get_mean()
        std = self.stat.get_std()

        score = np.divide(
            (X[0] - mean), std, out=np.zeros_like(X[0]), where=std != 0
        )

        return score