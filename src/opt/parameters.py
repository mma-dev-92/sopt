import dataclasses
import numpy as np

from typing import Self

from src.opt.indices import Indices
from src.preprocess.model import StorageParams, InputData


@dataclasses.dataclass
class Parameters:
    prices: np.ndarray
    dt: float
    storage_params: StorageParams

    @classmethod
    def create(cls, input_data: InputData, indices: Indices) -> Self:
        return cls(
            prices=input_data.market_data.prices.loc[indices.t_idx.vals, :].values,
            dt=input_data.market_data.resolution,
            storage_params=input_data.storage_params,
        )
