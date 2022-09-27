from os.path import dirname, join
from typing import Union

import numpy as np
import pandas as pd


class DS:
    def __init__(self) -> None:
        self.data = None
        self.date = None
        self.label = None
        self.features = None
        self.names = None
        self.path = None

    def preprocess(self) -> None:
        self.preprocess_data()
        self.preprocess_timestamp()
        self.preprocess_label()
        self.preprocess_feature()

    def preprocess_data(self) -> None:
        if type(self.path) == str:
            try:
                self.data = pd.read_csv(self.path)
            except FileExistsError:
                print("Cannot read this file:", self.path)
        elif type(self.path) == np.ndarray:
            self.data = pd.DataFrame(self.path)
        elif type(self.path) == pd.DataFrame:
            self.data = self.path
        self.names = self.data.columns.values

    def preprocess_timestamp(self) -> None:
        if "timestamp" in self.names.tolist():
            self.date = self.data["timestamp"].values
        elif "ts" in self.names.tolist():
            self.date = self.data["ts"].values
        else:
            self.date = self.data.index.values

    def preprocess_label(self) -> None:
        if "label" in self.names.tolist():
            self.label = np.array(self.data["label"].values)
        elif "is_anomaly" in self.names.tolist():
            self.label = np.array(self.data["is_anomaly"].values)

    def preprocess_feature(self) -> None:
        self.features = np.setdiff1d(
            self.names, np.array(["label", "is_anomaly", "timestamp", "ts"])
        )
        self.data = np.array(self.data[self.features])


class UnivariateDataSet(DS):

    def __init__(self, filepath: Union[str, np.ndarray, pd.DataFrame]) -> None:
        super().__init__()
        # module_path = dirname(__file__)
        # self.path = join(module_path, filename)
        self.path = filepath
        self.value = None
        self.preprocess()
        self.preprocess_value()

    def preprocess_value(self):
        if type(self.data) == np.ndarray:
            self.value = self.data.reshape(-1)
        else:
            self.value = self.data["value"].values
