from typing import Tuple

from loguru import logger


class HotSpotCuboid:
    def __init__(self, attribute_names: Tuple[str]):
        self.attribute_names = sorted(attribute_names)

        self._finished = False

    def run(self):
        if self._finished:
            logger.warning(f"the searching is over, for trying to rerun {self}")
            return
        logger.info(f"searching in {self} ....")
        self._get_sorted_elements_and_root()

    def _get_sorted_elements_and_root(self):
        pass


def __str__(self):
        return f"Cuboid {self.attribute_names}"
