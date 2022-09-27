import os

import pandas as pd
import numpy as np
import re


class MDDataSetLoader:

    def __init__(self, data_file_path: str):
        self.data_file_path = data_file_path
        self.data_df = None
        self.inject_info_df = None

    def load(self):
        df = None
        data_files, inject_file = self._list_data_files()
        # load the dimension/metrics data
        assert len(data_files) > 0, "can't read the data file"
        for csv_file in data_files:
            tmp_df = pd.read_csv(csv_file)
            tmp_df["timestamp"] = csv_file[-14:-4]

            if df is None:
                df = tmp_df
            else:
                df = pd.concat([df, tmp_df])
        self.data_df = df
        self.data_df['real'] = self.data_df['real'].astype(np.float64)
        self.data_df['predict'] = self.data_df['predict'].astype(np.float64)
        # load the inject info
        if inject_file is not None and os.path.exists(inject_file):
            self.inject_info_df = pd.read_csv(inject_file)
        return self

    def _list_data_files(self, is_number_name=True):
        res = []
        inject_file = None
        path = self.data_file_path
        for f in os.listdir(path):
            if f == 'injection_info.csv':
                inject_file = os.path.join(self.data_file_path, f)
            elif not re.match(r'\d*\.csv', f):
                continue
            else:
                file_abs_path = os.path.join(self.data_file_path, f)
                if os.path.isfile(file_abs_path):
                    res.append(file_abs_path)
        return res, inject_file

    @property
    def data(self):
        return self.data_df
    
    @property
    def inject_info(self):
        return self.inject_info_df
