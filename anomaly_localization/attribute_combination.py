import copy
import numpy as np
import pandas as pd


class AttributeCombination(dict):
    ANY = '__ANY__'

    def __init__(self, **kwargs):
        super().__init__(**{key: str(value) for key, value in kwargs.items()})
        self.__id = None
        self.non_any_keys = tuple()
        self.non_any_values = tuple()
        self.__is_terminal = False
        self._update()

    def __eq__(self, other: 'AttributeCombination'):
        return self.__id == other.__id

    def __lt__(self, other):
        return self.__id < other.__id

    def __le__(self, other):
        return self.__id <= other.__id

    def __hash__(self):
        return hash(self.__id)

    def __setitem__(self, key, value):
        super().__setitem__(key, str(value))
        self._update()

    def __str__(self):
        return "&".join(f"{key}={value}" for key, value in zip(self.non_any_keys, self.non_any_values))

    def _update(self):
        self.__id = tuple((key, self[key]) for key in sorted(self.keys()))
        self.non_any_keys = tuple(_ for _ in sorted(self.keys()) if self[_] != self.ANY)
        self.non_any_values = tuple(self[_] for _ in sorted(self.keys()) if self[_] != self.ANY)
        self.__is_terminal = not any(self.ANY == value for value in self.values())

    def copy_and_update(self, other):
        o = copy.copy(self)
        o.update(other)
        o._update()
        return o

    @classmethod
    def get_root_attribute_combination(cls, attribute_names):
        return AttributeCombination(**{key: AttributeCombination.ANY for key in attribute_names})

    def index_dataframe(self, data: pd.DataFrame):
        if len(self.non_any_values) == 0:
            return np.ones(len(data), dtype=np.bool)
        try:
            arr = np.zeros(shape=len(data), dtype=np.bool)
            if "loc" not in data.columns:
                if len(self.non_any_values) == 1:
                    idx = data.index.get_loc(self.non_any_values[0])
                else:
                    idx = data.index.get_loc(self.non_any_values)
            else:
                idx = np.array(data.loc[self.non_any_values]['loc'], dtype=int)
            arr[idx] = True
            return arr
        except KeyError:
            return np.zeros(len(data), dtype=np.bool)
