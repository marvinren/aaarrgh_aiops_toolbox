from abc import ABC, abstractmethod
from typing import Optional, Any, Sequence

import pandas as pd

from aaarrgh.dataset import TimeSeriesDataSet


class AnomalyPoint:

    def __init__(self, start_time: pd.Timestamp, end_time: pd.Timestamp, confidence: float):
        self.start_time = start_time
        self.end_time = end_time
        self.confidence = confidence

    def __repr__(self):
        return f"AnomalyPoint({self.start_time}-{self.end_time}) Conf Scores: {self.confidence}"


class TimeSeriesAnomalyDetector(ABC):

    def __init__(self):
        self.__type__ = "time_series_anomaly_detector"

    @abstractmethod
    def detect(self, data: TimeSeriesDataSet) -> Sequence[AnomalyPoint]:
        raise NotImplementedError()


class AnomalyResponse:

    def __init__(self, labels, scores) -> None:
        self.labels = labels
        self.scores = scores


class TimeSeriesAnomalyModel(ABC):

    @abstractmethod
    def __init__(self, serialized_model: Optional[bytes]) -> None:
        raise NotImplementedError()

    @abstractmethod
    def serialize(self) -> bytes:
        raise NotImplementedError()

    @abstractmethod
    def fit(
            self,
            data: TimeSeriesDataSet,
            historical_data: Optional[TimeSeriesDataSet],
            **kwargs: Any,
    ) -> None:
        raise NotImplementedError()

    @abstractmethod
    def predict(
            self,
            data: TimeSeriesDataSet,
            historical_data: Optional[TimeSeriesDataSet],
            **kwargs: Any,
    ) -> AnomalyResponse:
        raise NotImplementedError()

    @abstractmethod
    def fit_predict(
            self,
            data: TimeSeriesDataSet,
            historical_data: Optional[TimeSeriesDataSet],
            **kwargs: Any,
    ) -> AnomalyResponse:
        raise NotImplementedError()
