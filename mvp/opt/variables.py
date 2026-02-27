import cvxpy as cp
import dataclasses

from typing import Self

from mvp.preprocess.model import InputData


@dataclasses.dataclass
class Variables:
    soc: cp.Variable
    gen: cp.Variable
    load: cp.Variable

    @classmethod
    def create(cls, input_data: InputData) -> Self:
        tt = len(input_data.market_data.prices)
        return cls(
            soc=cp.Variable(shape=(tt, 1), name='soc', nonneg=True),
            gen=cp.Variable(shape=(tt, 1), name='gen', nonneg=True),
            load=cp.Variable(shape=(tt, 1), name='load', nonneg=True),
        )
