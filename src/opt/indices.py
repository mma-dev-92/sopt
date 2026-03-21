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

    @classmethod
    def create(cls, input_data: InputData) -> Self:
        return cls(
            t_idx=cls.create_time_index(input_data),
        )

    @staticmethod
    def create_time_index(input_data: InputData) -> Index:
        start = input_data.params.timestep_start
        end = input_data.params.timestep_end
        dt = input_data.market_data.resolution

        min_resolution = int(_MINUTES_PER_HOUR * dt)

        timesteps = pd.date_range(
            start=pd.Timestamp(start),
            end=pd.Timestamp(end) - pd.Timedelta(minutes=min_resolution),
            freq=f'{min_resolution}min',
        )
        return Index(name='timestep', values=timesteps.values)