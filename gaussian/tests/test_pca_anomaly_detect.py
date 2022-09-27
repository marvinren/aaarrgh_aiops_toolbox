#!/usr/bin/env python
# -*- coding: utf-8 -*-
import unittest
import logging

from gaussian.loganomaly.model.PCA import PCALogAnomaly

logging.basicConfig(level=logging.DEBUG)
from gaussian.loganomaly import dataloader, preprocessing


class TestPcaAnomalyDetect(unittest.TestCase):
    def test_pca_anomaly_detect(self):
        struct_log = "/Users/renzhiqiang/Workspace/aiops/log/structured_hdfs_log/HDFS_100k.log_structured.csv"
        label_file = "/Users/renzhiqiang/Workspace/aiops/log/structured_hdfs_log/anomaly_label.csv"
        (x_train, y_train), (x_test, y_test) = dataloader.load_HDFS(struct_log, label_file=label_file,
                                                                    window='session', train_ratio=0.75, split_type='uniform')

        feature_extractor = preprocessing.FeatureExtractor()
        x_train = feature_extractor.fit_transform(x_train, term_weighting='tf-idf',
                                                  normalization='zero-mean')
        x_test = feature_extractor.transform(x_test)

        model = PCALogAnomaly()
        model.fit(x_train)

        print('Train validation:')
        precision, recall, f1 = model.evaluate(x_train, y_train)

        print('Test validation:')
        precision, recall, f1 = model.evaluate(x_test, y_test)
