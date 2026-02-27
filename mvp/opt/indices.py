import dataclasses
import numpy as np

from typing import Self

from mvp.preprocess.model import InputData


class Index:

    def __init__(self, name: str, values: np.ndarray) -> None:

        assert np.unique(values).size == values.size, 'indexing set must contain unique elements'

        self.name = name
        self._values = values.squeeze()
        self._ii = np.arange(len(values.squeeze()))

    @property
    def ii(self) -> np.ndarray:
        return self._ii

    @property
    def vals(self) -> np.ndarray:
        return self._values


@dataclasses.dataclass
class Indices:

    t_idx: Index

    @classmethod
    def create(cls, input_data: InputData) -> Self:
        return cls(
            t_idx=Index(name='T', values=input_data.market_data.prices.index),
        )