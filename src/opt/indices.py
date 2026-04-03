import dataclasses
import pandas as pd
import numpy as np

from typing import Self

from src.preprocess.model import InputData


_MINUTES_PER_HOUR = 60

class Index:

    def __init__(self, name: str, values: np.ndarray) -> None:

        assert np.unique(values).size == values.size, 'indexing set must contain unique elements'

        self.name = name
        self._values = values
        self._ii = np.arange(len(values))

    @property
    def ii(self) -> np.ndarray:
        return self._ii

    @property
    def vals(self) -> np.ndarray:
        return self._values


@dataclasses.dataclass
class Indices:
    t_idx: Index
    """timestep index (sequence of consecutive natural numbers)"""

    @classmethod
    def create(cls, input_data: InputData, n_hours: int = 24) -> Self:
        dt = input_data.market_data.resolution
        tt = int(n_hours/ dt)
        return cls(
            t_idx=Index(name='timestep', values=np.arange(tt)),
        )
