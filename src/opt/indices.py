import dataclasses
import numpy as np

from typing import Self



_MINUTES_PER_HOUR = 60

class Index:

    def __init__(self, name: str, values: np.ndarray) -> None:

        assert np.unique(values).size == values.size, 'indexing set must contain unique elements'

        self.name = name
        self._values = values
        self._ii = np.arange(len(values))

    def __len__(self) -> int:
        return len(self._values)

    @property
    def ii(self) -> np.ndarray:
        return self._ii

    @property
    def vals(self) -> np.ndarray:
        return self._values


class TimeIndex(Index):

    def __init__(self, dt: float, n_hours: int) -> None:
        super().__init__(name="TimeIndex", values=np.arange(int(n_hours * dt)))
        self._dt = dt
        self._n_hours = n_hours

    @property
    def dt(self) -> float:
        """
        Time resolution (fraction of hour)

        if dt = 0.25 then time resolution is 15-minutes
        """
        return self._dt

    @property
    def n_hours(self) -> int:
        """
        Number of hours in TimeIndex
        """
        return self._n_hours
