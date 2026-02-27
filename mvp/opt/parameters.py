import dataclasses
import numpy as np

from typing import Self

from mvp.preprocess.model import StorageParams, InputData


@dataclasses.dataclass
class Parameters:
    prices: np.ndarray
    storage_params: StorageParams

    @classmethod
    def create(cls, input_data: InputData) -> Self:
        return cls(
            prices=input_data.market_data.prices.values,
            storage_params=input_data.storage_params,
        )
