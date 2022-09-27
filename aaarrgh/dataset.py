import pandas as pd
import logging
from typing import Union, Optional, List
import matplotlib.pyplot as plt

DEFAULT_TIME_NAME = "time"
DEFAULT_VALUE_NAME = "value"


def _log_error(msg: str) -> ValueError:
    logging.error(msg)
    return ValueError(msg)


class TimeSeriesDataSet:

    def __init__(self,
                 df: pd.DataFrame = None,
                 time_col_name: str = DEFAULT_TIME_NAME,
                 value_col_name: Union[str, list] = DEFAULT_VALUE_NAME,
                 unix_time_units: str = "s",
                 parse_time_col: bool = False):
        self.time_col_name = time_col_name
        self.value_col_name = value_col_name

        if df is None:
            raise _log_error("Argument df needs to be a pandas.DataFrame, and be not None")
        if self.time_col_name not in df.columns:
            msg = f"Time column {self.time_col_name} not in DataFrame"
            raise _log_error(msg)

        if parse_time_col:
            df[self.time_col_name] = pd.to_datetime(self.time_col_name, unit=unix_time_units)

        df.index = df[self.time_col_name]
        # df.sort_values(self.time_col_name, inplace=True)
        # df.reset_index(inplace=True, drop=True)
        # df.drop(columns=[self.time_col_name], axis=1, inplace=True)

        self._time = df[self.time_col_name]

        if isinstance(self.value_col_name, str):
            if self.value_col_name not in df.columns:
                msg = f"Time column {self.value_col_name} not in DataFrame"
                raise _log_error(msg)
            self._is_univariate = True
            self._value = df[self.value_col_name]
        elif isinstance(self.value_col_name, list):
            for v_col in self.value_col_name:
                if v_col not in df.columns:
                    msg = f"Time column {self.value_col_name} not in DataFrame"
                    raise _log_error(msg)
            self._is_univariate = False
            self._value = df[self.value_col_name]

    def resample_interpolate(self, resample_freq: str = "5T"):
        df = self.to_dataframe()
        df = df.resample(resample_freq).mean().ffill()
        self._time = df.index
        if isinstance(self.value_col_name, str):
            if self.value_col_name not in df.columns:
                msg = f"Time column {self.value_col_name} not in DataFrame"
                raise _log_error(msg)
            self._value = df[self.value_col_name]
        elif isinstance(self.value_col_name, list):
            for v_col in self.value_col_name:
                if v_col not in df.columns:
                    msg = f"Time column {self.value_col_name} not in DataFrame"
                    raise _log_error(msg)
            self._value = df[self.value_col_name]

    @property
    def is_univariate(self):
        return self._is_univariate

    def to_dataframe(self, standard_time_col_name: bool = False) -> pd.DataFrame:

        time_col_name = (
            DEFAULT_TIME_NAME if standard_time_col_name else self.time_col_name
        )
        output_df = pd.DataFrame(dict(zip((time_col_name,), (self._time,))), copy=False)
        if isinstance(self._value, pd.Series):
            if self._value.name is not None:
                output_df[self._value.name] = self._value.values
            else:
                output_df[DEFAULT_VALUE_NAME] = self._value
        elif isinstance(self._value, pd.DataFrame):
            output_df = pd.concat(
                [output_df, self._value], axis=1, copy=False
            ).reset_index(drop=True)
        else:
            raise ValueError(f"Wrong value type: {type(self.value)}")

        return output_df


class TimeSeriesDataSetPlot:

    def __init__(self, ts: TimeSeriesDataSet):
        self.ts = ts

    def plot(self, cols: Optional[List[str]] = None, ):
        df = self.ts.to_dataframe()

        all_cols = list(df.columns)
        all_cols.remove(self.ts.time_col_name)

        if cols is None:
            cols = all_cols
        elif not set(cols).issubset(all_cols):
            logging.error(f"Columns to plot: {cols} are not all in the timeseries")
            raise ValueError("Invalid columns passed")

        fig = plt.figure(figsize=(13, 5))
        ax = fig.gca()
        df.plot(x=self.ts.time_col_name, y=cols, ax=ax)
        plt.show()
