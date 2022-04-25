import os.path

import pandas as pd



class ServiceMetrics:

    def __init__(self):
        self._data = None

    def load_from_file(self, local_file: str) :
        if not os.path.exists(local_file):
            raise Exception("can't find the data file {}".format(self.local_file))
        df = pd.read_csv(local_file)
        if self._data is None:
            self._data = df
        else:
            self._data = df.add(self._data)
        return self

    def load(self):
        return self

    @property
    def data(self):
        return self._data


class HostMetrics:

    def __init__(self):
        self._data = None

    def load_from_file(self, local_file: str):
        if not os.path.exists(local_file):
            raise Exception("can't find the data file {}".format(self.local_file))
        self._data = pd.read_csv(local_file)
        return self

    def load(self):
        return self

    @property
    def data(self):
        return self._data
