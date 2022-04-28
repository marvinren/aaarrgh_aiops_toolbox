from typing import Generator, Optional, Union

import numpy as np
import pandas as pd


class AnomalyUnivariateGen:
    def __init__(self, n=2000, anomaly_prob=0.05):
        self.n = n
        self.anomaly_prob = anomaly_prob

    def generate(self):
        n = self.n
        x = np.sin(2 * np.pi * 1 * np.linspace(0, n / 200, n))
        x_rand = np.random.rand(x.size)
        prob = np.random.uniform(0, 1, x.size)

        x_rand_list = []
        labels = []
        for item, item_probability in zip(x_rand, prob):
            if item_probability > 1. - self.anomaly_prob:
                x_rand_list.append(item * item_probability * 10)
                labels.append(-1)
            else:
                x_rand_list.append(item)
                labels.append(1)

        x += np.array(x_rand_list)
        labels = np.array(labels)
        return x, labels


# class StreamGenerator:
#
#     def __init__(self,
#                  X: Union[pd.DataFrame, np.ndarray],
#                  y: Optional[Union[pd.Series, np.ndarray]] = None,
#                  features: list = None,
#                  shuffle: bool = False, ):
#         if y is not None:
#             assert len(X) == len(y)
#
#         if isinstance(X, np.ndarray):
#             self.X = X
#             self.y = [None] * len(X) if y is None else y
#         elif isinstance(X, pd.DataFrame):
#             self.X = X.to_numpy() if features == None else X[features].to_numpy()
#             self.y = [None] * len(X) if y is None else y.to_numpy()
#         else:
#             raise TypeError(
#                 "Unexpected input data type, except np.ndarray or pd.DataFrame"
#             )
#         self.features = features
#         self.index = list(range(len(X)))
#         if shuffle:
#             np.random.shuffle(self.index)
#
#     def iter_item(self) -> Generator:
#         for i in self.index:
#             yield self.X[i], self.y[i]
#
#     def get_features(self) -> list:
#         return self.features

if __name__ == "__main__":
    a = AnomalyUnivariateGen()
    data, label = a.generate()
    import matplotlib.pyplot as plt
    plt.scatter(range(0, data.size), data, c=label, s=2)
    plt.show()
