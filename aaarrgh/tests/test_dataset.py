import unittest
import pandas as pd
import numpy as np
from aaarrgh.dataset import TimeSeriesDataSet, TimeSeriesDataSetPlot


class TestTimeSeriesData(unittest.TestCase):
    def test_dataframe_univariate_TimeSeries(self):
        df = pd.DataFrame(
            {
                "time": [
                    "2022-01-01 00:00:00",
                    "2022-01-01 00:01:00",
                    "2022-01-01 00:02:00",
                ],
                "value": [1, 2, 3],
            }
        )

        ts = TimeSeriesDataSet(df)
        print(ts.to_dataframe().head())
        assert ts.is_univariate

    def test_dataframe_multivariate_TimeSeries(self):
        df = pd.DataFrame(
            {
                "time": [
                    "2022-01-01 00:00:00",
                    "2022-01-01 00:01:00",
                    "2022-01-01 00:02:00",
                ],
                "v1": [1, 2, 3],
                "v2": [4, 5, 6],
                "v3": [7, 8, 9],
            }
        )

        ts = TimeSeriesDataSet(df, value_col_name=["v1", "v2", "v3"])
        print(ts.to_dataframe().head())

        assert not ts.is_univariate

    def test_dataframe_resample_inter(self):
        n = 1000
        df = pd.DataFrame(
            {
                "time": pd.date_range("20220101120000", periods=n, freq="1T"),
                "value": np.sin(np.linspace(0, 2 * np.pi * 10, 1000))
                + np.random.random(n),
            }
        )
        ts = TimeSeriesDataSet(df)
        ts.resample_interpolate()
        df = ts.to_dataframe()
        assert "value" in df.columns

        # pl = TimeSeriesDataSetPlot(ts)
        # pl.plot(cols=["value"])

    def test_dataframe_variable_TimeSeries(self):
        n = 1000
        df = pd.DataFrame(
            {
                "time": pd.date_range("20220101120000", periods=n, freq="1T"),
                "value": np.sin(np.linspace(0, 2 * np.pi * 20, 1000))
                + np.random.random(n),
            }
        )
        ts = TimeSeriesDataSet(df)
        pl = TimeSeriesDataSetPlot(ts)
        pl.plot(cols=["value"])
