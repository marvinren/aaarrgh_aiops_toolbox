import unittest
import pandas as pd
import numpy as np

from aaarrgh.anomaly.stat_anomaly_detector import StatisticsAnomalyDetector
from aaarrgh.dataset import TimeSeriesDataSet


class TestStatisticsAnomalyDetector(unittest.TestCase):

    def test_three_sigma_anomaly_detect(self):
        n = 1000
        df = pd.DataFrame(
            {
                "time": pd.date_range("20220101120000", periods=n, freq='1T'),
                "value": np.sin(np.linspace(0, 2 * np.pi * 10, 1000)) + np.random.random(n)
            }
        )
        df.loc[n-1, "value"] = 100
        df.loc[n-2, "value"] = -100
        ts = TimeSeriesDataSet(df)

        detector = StatisticsAnomalyDetector()
        ret = detector.detect(ts)
        print(ret)
        assert len(ret) == 2
