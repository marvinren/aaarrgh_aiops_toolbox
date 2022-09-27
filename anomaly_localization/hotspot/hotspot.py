from itertools import combinations
from typing import Callable, FrozenSet, Union, List
import pandas as pd
import numpy as np

from anomaly_localization.attribute_combination import AttributeCombination
from anomaly_localization.base import _BaseAnomalyLocalization
from anomaly_localization.data import MDDataSetLoader
from loguru import logger
from heapq import merge


class HotSpot(_BaseAnomalyLocalization):

    def __init__(self, op: Callable = lambda x: x, max_steps=None, max_time=None, ps_upper_threshold=1.0,
                 ps_lower_threshold=0.0):
        self.op = op
        self.ps_lower_threshold = ps_lower_threshold if ps_lower_threshold is not None else float('-inf')
        self.max_time = max_time
        self.max_steps = max_steps
        self.ps_upper_threshold = ps_upper_threshold if ps_upper_threshold is not None else float('+inf')
        self.K = 3

        self.attribute_names = None
        self.attribute_values = None
        self._best_nodes = list()
        self._finished = False
        self.cuboids = [[]]

    def fit_predict(self, X: pd.DataFrame):
        assert len(X.columns) >= 4, f"the demension must be equal to or greater than 2, except the [real] and [predict]"
        assert 'real' in X.columns, f"the columns of data X must contain [real], {X.columns}"
        assert 'predict' in X.columns, f"the columns of data X must contain [predict], {X.columns}"
        logger.info(f"get the attribute names/values.")
        X.real = X.real.astype(np.float64)
        X.predict = X.predict.astype(np.float64)
        cols = list(set(X.columns) - {'real', 'predict'})
        attribute_values = {}
        for col in cols:
            attribute_values[col] = list(set(X[col]))
        self.attribute_names = cols
        self.attribute_values = attribute_values
        assert len(self.attribute_names) > 0, f"there is no available attributes, {self.attribute_names}"
        logger.info(f'available attribute names: {self.attribute_names}.')

        if self._finished:
            logger.warning(f"the searching is over, for trying to rerun {self}")
            return
        logger.info(f"searching in {self} ....")
        logger.info(f"start mcts searching...")
        for layer_i in np.arange(len(self.attribute_names)) + 1:
            logger.info(f"search Layer_{layer_i}")
            self.cuboids.append([])
            for cuboid in combinations(self.attribute_names, layer_i):
                logger.info(f"MCTS in {cuboid} of layer_{layer_i}")
                # TODO: MCTS in cuboid
                hotspot_cuboid = HotSpotCuboidMCTS(self.attribute_names, self.attribute_values, cuboid, self.cuboids, X)
                hotspot_cuboid.search()
                if hotspot_cuboid is not None:
                    self.cuboids[layer_i].append(hotspot_cuboid)
            for cuboid in self.cuboids[layer_i]:
                _best_nodes = [node for node in cuboid.best_nodes]
                logger.info(f"find the best node in layer_{layer_i}: {_best_nodes}")
                # for succinctness (Occam's razor principle)
                # merge the global best-nodes and the layer-level best-nodes
                self._best_nodes = list(merge(self._best_nodes, cuboid.best_nodes,
                                              key=lambda x: x.result_score, reverse=True))[:self.K]
                # stop searching, when ps score >= PT
                if len(self._best_nodes) > 0 and self._best_nodes[0].potential_score >= self.ps_upper_threshold:
                    break
            else:
                continue
            break
        # sort the root cause node by result_score(?)
        self._best_nodes = sorted(self._best_nodes, key=lambda x: x.result_score, reverse=True)
        self._finished = True

    @property
    def best_nodes(self):
        assert self._finished, 'HotSpot has not run.'
        return self._best_nodes

    # def searchbyMCTS(self, layer_i, cuboid):
    #     logger.info(f"start search in {cuboid} by MCTS")
    #     root_attribute_combination = AttributeCombination.get_root_attribute_combination(self.attribute_names)
    #     self._layer = layer_i
    #     self._attribute_names = self.attribute_names
    #     self._root = HotSpotNode(root_attribute_combination)
    #     self._get_sorted_elements_and_root(root_attribute_combination)
    #     return None
    #
    # def _get_sorted_elements_and_root(self, root_attribute_combination):
    #     if self._layer == 1:
    #         values = self.attribute_values[self.attribute_names[0]]
    #         self.elements = [
    #             HotSpotElement(frozenset({root_attribute_combination.copy_and_update(
    #                 {self.attribute_names[0]: value}
    #             )})
    #             ) for value in values
    #         ]
    #     else:
    #         self.elements = []
    #         for cuboid in self.hotspot.cuboids[self.layer - 1]:
    #             pass


class HotSpotCuboidMCTS:
    def __init__(self, root_attribute_names, root_attribute_values, attribute_names, cuboids, data):
        self._root_attribute_names = root_attribute_names
        self._root_attribute_values = root_attribute_values
        self._attribute_names = sorted(attribute_names)
        self._layer = len(self._attribute_names)
        self._finished = False
        self._root = False
        self.elements = None
        self._best_nodes = []
        self._root_cuboids = cuboids
        self._data = data
        self._indexed_data = list(self._data.set_index(self._attribute_names).sort_index())

    @property
    def best_nodes(self) -> List['HotSpotNode']:
        # assert self._finished, f'{self._attribute_names} search not complete'
        return self._best_nodes

    def search(self):
        if self._finished:
            logger.warning(f"This Cuboid {self._attribute_names} has been searched. Don't try it again.")
            return
        logger.info(f"start search in Cuboid {self._attribute_names} by MCTS")
        self._get_sorted_elements_and_root()

        # TODO: not completed

    def _get_sorted_elements_and_root(self):
        root_attribute_combination = AttributeCombination.get_root_attribute_combination(self._root_attribute_names)
        self._root = HotSpotNode(None, frozenset({root_attribute_combination}))
        if self._layer == 1:
            values = self._root_attribute_values[self._attribute_names[0]]
            # self.elements = [
            #     HotSpotElement(
            #         frozenset({root_attribute_combination.copy_and_update({self._attribute_names[0]: value})}),
            #         self._indexed_data) for
            #     value in values
            # ]
            self.elements = [
                {root_attribute_combination.copy_and_update({self._attribute_names[0]: value})}for value in values
            ]
        else:
            self.elements = []
            for cuboid in self._root_cuboids[self._layer - 1]:
                pass
            # TODO: not completed
        print(self.elements)
        print(self._indexed_data)
        self.elements = sorted(list(set(self.elements)), key=lambda x: x.potential_score, reverse=True)
        logger.info(f"number of element after  filter: {len(self.elements)}")


class HotSpotNode:
    def __init__(self, parent: Union['HotSpotNode', None], attribute_combinations: FrozenSet[AttributeCombination]):
        self.parent = parent
        self.attribute_combinations = attribute_combinations


class HotSpotElement:
    def __init__(self, attribute_combinations: FrozenSet[AttributeCombination], indexed_data):
        self.attribute_combinations = attribute_combinations
        self.indexed_data = indexed_data
        self._update_ps_score()

    def _update_ps_score(self):
        _data_index = list(AttributeCombination.index_dataframe(
            self.attribute_combinations,
            data
        ) for data in self.indexed_data)


if __name__ == "__main__":
    logger.info("load the local data")
    data_path = "/Users/renzhiqiang/Workspace/data/Psqueeze-dataset/A/A_week_12_cuboid_layer_1_n_ele_1"
    data_loader = MDDataSetLoader(data_file_path=data_path).load()
    df = data_loader.data
    df_eval = data_loader.inject_info
    logger.info("get the first timestamp cuboid")
    df = df[df['timestamp'] == df['timestamp'].unique()[0]]
    df = df.drop(['timestamp'], axis=1)
    logger.info(df.head())
    logger.info("search root cause set by hotspot")
    model = HotSpot(max_steps=100, ps_upper_threshold=0.95, ps_lower_threshold=0.95)
    model.fit_predict(df)
