from collections import OrderedDict

import numpy as np
import pandas as pd
from sklearn.utils import shuffle
import regex as re
import logging


def _split_data(x_data, y_data=None, train_ratio=0, split_type='uniform'):
    if split_type == 'uniform' and y_data is not None:
        pos_idx = y_data > 0
        x_pos = x_data[pos_idx]
        y_pos = y_data[pos_idx]
        x_neg = x_data[~pos_idx]
        y_neg = y_data[~pos_idx]
        train_pos = int(train_ratio * x_pos.shape[0])
        train_neg = int(train_ratio * x_neg.shape[0])
        x_train = np.hstack([x_pos[0:train_pos], x_neg[0:train_neg]])
        y_train = np.hstack([y_pos[0:train_pos], y_neg[0:train_neg]])
        x_test = np.hstack([x_pos[train_pos:], x_neg[train_neg:]])
        y_test = np.hstack([y_pos[train_pos:], y_neg[train_neg:]])
    elif split_type == 'sequential':
        num_train = int(train_ratio * x_data.shape[0])
        x_train = x_data[0:num_train]
        x_test = x_data[num_train:]
        if y_data is None:
            y_train = None
            y_test = None
        else:
            y_train = y_data[0:num_train]
            y_test = y_data[num_train:]
    # Random shuffle
    indexes = shuffle(np.arange(x_train.shape[0]))
    x_train = x_train[indexes]
    if y_train is not None:
        y_train = y_train[indexes]
    return (x_train, y_train), (x_test, y_test)


def load_HDFS(log_file, label_file=None, window='session', train_ratio=0.75, split_type="sequential", save_csv=False,
              window_size=0):
    if log_file.endswith('.npz'):
        # Split training and validation set in a class-uniform way
        data = np.load(log_file)
        x_data = data['x_data']
        y_data = data['y_data']
        (x_train, y_train), (x_test, y_test) = _split_data(x_data, y_data, train_ratio, split_type)

    elif log_file.endswith('.csv'):
        logging.info("读取HDFS文件: %s", log_file)
        struct_log = pd.read_csv(log_file, engine='c',
                                 na_filter=False, memory_map=True)
        data_dict = OrderedDict()
        for idx, row in struct_log.iterrows():
            blkId_list = re.findall(r'(blk_-?\d+)', row['Content'])
            blkId_set = set(blkId_list)
            for blk_Id in blkId_set:
                if not blk_Id in data_dict:
                    data_dict[blk_Id] = []
                data_dict[blk_Id].append(row['EventId'])
        data_df = pd.DataFrame(list(data_dict.items()), columns=['BlockId', 'EventSequence'])
        logging.info("读取日志文件完成，共有%d个block", len(data_dict))

        if label_file:
            # Split training and validation set in a class-uniform way
            label_data = pd.read_csv(label_file, engine='c', na_filter=False, memory_map=True)
            label_data = label_data.set_index('BlockId')
            label_dict = label_data['Label'].to_dict()
            data_df['Label'] = data_df['BlockId'].apply(lambda x: 1 if label_dict[x] == 'Anomaly' else 0)

            # Split train and test data
            (x_train, y_train), (x_test, y_test) = _split_data(data_df['EventSequence'].values,
                                                               data_df['Label'].values, train_ratio, split_type)
            logging.info("标注文件获取的训练有%d条，测试有%d条", y_train.sum(), y_test.sum())
        if save_csv:
            data_df.to_csv('data_instances.csv', index=False)
            logging.info("存入结果文件: data_instances.csv")

        if window_size > 0:
            x_train, window_y_train, y_train = slice_hdfs(x_train, y_train, window_size)
            x_test, window_y_test, y_test = slice_hdfs(x_test, y_test, window_size)
            log = "{} {} windows ({}/{} anomaly), {}/{} normal"
            logging.info(log.format("Train:", x_train.shape[0], y_train.sum(), y_train.shape[0], (1 - y_train).sum(),
                                    y_train.shape[0]))
            logging.info(log.format("Test:", x_test.shape[0], y_test.sum(), y_test.shape[0], (1 - y_test).sum(),
                                    y_test.shape[0]))
            return (x_train, window_y_train, y_train), (x_test, window_y_test, y_test)

        if label_file is None:
            if split_type == 'uniform':
                split_type = 'sequential'
                logging.info('Warning: Only split_type=sequential is supported \
                        if label_file=None.'.format(split_type))
            # Split training and validation set sequentially
            x_data = data_df['EventSequence'].values
            (x_train, _), (x_test, _) = _split_data(x_data, train_ratio=train_ratio, split_type=split_type)
            logging.info('Total: {} instances, train: {} instances, test: {} instances'.format(
                x_data.shape[0], x_train.shape[0], x_test.shape[0]))
            return (x_train, None), (x_test, None), data_df
    else:
        raise NotImplementedError('load_HDFS() only support csv and npz files!')

    num_train = x_train.shape[0]
    num_test = x_test.shape[0]
    num_total = num_train + num_test
    num_train_pos = sum(y_train)
    num_test_pos = sum(y_test)
    num_pos = num_train_pos + num_test_pos

    logging.info("一共有%d个实例，%d个异常，%d个正常", num_total, num_pos, num_total - num_pos)
    logging.info("训练集，有%d个实例，%d个异常，%d个正常", num_train, num_train_pos, num_train - num_train_pos)
    logging.info("测试集，有%d个实例，%d个异常，%d个正常", num_test, num_test_pos, num_test - num_test_pos)

    return (x_train, y_train), (x_test, y_test)


def slice_hdfs(x, y, window_size):
    results_data = []
    logging.info("Slicing {} sessions, with window {}".format(x.shape[0], window_size))
    for idx, sequence in enumerate(x):
        seqlen = len(sequence)
        i = 0
        while (i + window_size) < seqlen:
            slice = sequence[i: i + window_size]
            results_data.append([idx, slice, sequence[i + window_size], y[idx]])
            i += 1
        else:
            slice = sequence[i: i + window_size]
            slice += ["#Pad"] * (window_size - len(slice))
            results_data.append([idx, slice, "#Pad", y[idx]])
    results_df = pd.DataFrame(results_data, columns=["SessionId", "EventSequence", "Label", "SessionLabel"])
    logging.info("Slicing done, {} windows generated".format(results_df.shape[0]))
    return results_df[["SessionId", "EventSequence"]], results_df["Label"], results_df["SessionLabel"]
