import cvxpy as cp
import dataclasses

from typing import Self

from mvp.opt.indices import Indices


@dataclasses.dataclass
class Variables:
    soc: cp.Variable
    gen: cp.Variable
    load: cp.Variable

    @classmethod
    def create(cls, indices: Indices) -> Self:
        tt = len(indices.t_idx.vals)
        return cls(
            soc=cp.Variable(tt, name='soc', nonneg=True),
            gen=cp.Variable(tt, name='gen', nonneg=True),
            load=cp.Variable(tt, name='load', nonneg=True),
        )
