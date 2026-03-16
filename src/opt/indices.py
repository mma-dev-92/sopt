import dataclasses
import pandas as pd
import numpy as np

from typing import Self
from datetime import date

from src.preprocess.model import InputData


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

    @classmethod
    def create(cls, input_data: InputData) -> Self:
        return cls(
            t_idx=cls.create_time_index_for_one_partition(input_data.params.partition),
        )

    @staticmethod
    def create_time_index_for_one_partition(partition: date) -> Index:
        """
        For now hardcoded - index for one partition (24h) in 15min freq

        To be generalized later
        """
        timestamps = pd.date_range(
            start=pd.Timestamp(partition),
            end=pd.Timestamp(partition) + pd.Timedelta(hours=23, minutes=45),
            freq='15min',
        )
        return Index(name='timestamp', values=timestamps.values)