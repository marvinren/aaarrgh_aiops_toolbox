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


if __name__ == "__main__":
    a = AnomalyUnivariateGen()
    data, label = a.generate()
    import matplotlib.pyplot as plt

    plt.scatter(range(0, data.size), data, c=label, s=2)
    plt.show()
