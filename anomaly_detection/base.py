from abc import ABC, abstractmethod
from typing import Union

import numpy as np
import pandas as pd


class BaseDetector(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def fit(self, X: Union[np.ndarray, pd.DataFrame]) -> None:
        return self

    @abstractmethod
    def score(self, X: Union[np.ndarray, pd.DataFrame]) -> float:
        return 0.0

    def fit_score(self, X: Union[np.ndarray, pd.DataFrame]) -> float:
        return self.fit(X).score(X)
