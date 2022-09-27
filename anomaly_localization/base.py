import abc

import pandas as pd


class _BaseAnomalyLocalization(metaclass=abc.ABCMeta):

    def __init__(self):
        pass

    @abc.abstractmethod
    def fit_predict(self, X:pd.DataFrame):
        return None

