import os
import unittest
import numpy as np
import pandas as pd

from anomaly_detection import AnomalyUnivariateGen
from anomaly_detection import UnivariateDataSet


class TestDataSet(unittest.TestCase):

    def test_univariate_dataset_by_ndarray(self):
        data = np.random.uniform(1, 100, 200)
        ds = UnivariateDataSet(data)

        self.assertEqual(len(ds.value), 200)
        self.assertGreaterEqual(ds.value.min(), 1)
        self.assertLessEqual(ds.value.max(), 100)

    def test_univariate_dataset_by_file(self):
        dirname = os.path.dirname(__file__)
        filepath = os.path.join(dirname, "data", "uniData.csv")
        ds = UnivariateDataSet(filepath)
        self.assertEqual(ds.data.shape, (1421, 1))
        self.assertEqual(len(ds.label), 1421)
        self.assertEqual(len(ds.features), 1)
        self.assertEqual(len(ds.value), 1421)
        self.assertEqual(len(ds.date), 1421)

    def test_univariate_data_gen(self):
        gen = AnomalyUnivariateGen()
        data, label = gen.generate()
        df = pd.DataFrame({"value": data, "label": label})
        ds = UnivariateDataSet(df)
        self.assertEqual(ds.data.shape, (2000, 1))


if __name__ == '__main__':
    unittest.main(verbosity=2)
