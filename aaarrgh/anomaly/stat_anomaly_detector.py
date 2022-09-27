from collections import Sequence
from typing import List

from aaarrgh.anomaly.detector import TimeSeriesAnomalyDetector, AnomalyPoint
from aaarrgh.dataset import TimeSeriesDataSet
import numpy as np


class StatisticsAnomalyDetector(TimeSeriesAnomalyDetector):

    def __init__(self, method: str = "sigma", sigma_index: int = 3) -> None:
        super().__init__()
        self.__type__ = "statistics"
        self.method = method
        self.sigma_index = sigma_index

    def detect(self, data: TimeSeriesDataSet) -> List[AnomalyPoint]:
        anomalies = []
        df = data.to_dataframe()
        anomalies_set = set()

        if self.method == "sigma":
            if data.is_univariate:
                up_threshold = np.mean(df[data.value_col_name]) + self.sigma_index * np.std(df[data.value_col_name])
                down_threshold = np.mean(df[data.value_col_name]) - self.sigma_index * np.std(df[data.value_col_name])
                anomalies_set = set(df[(df[data.value_col_name] > up_threshold) | (df[data.value_col_name] < down_threshold)].index)
            else:
                for col in df.columns:
                    up_threshold = np.mean(df[col]) + self.sigma_index * np.std(df[col])
                    down_threshold = np.mean(df[col]) - self.sigma_index * np.std(df[col])
                    anomalies_set.union(set(df[df[col] > up_threshold | df[col] < down_threshold].time))
            for a in anomalies_set:
                anomalies.append(AnomalyPoint(a, a, 1))

        return anomalies
