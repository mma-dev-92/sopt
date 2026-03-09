import cvxpy as cp
import dataclasses

from typing import Self

from src.opt.indices import Indices


@dataclasses.dataclass
class Variables:
    # decision variables
    soc: cp.Variable
    """storage state of charge"""
    gen: cp.Variable
    """storage generation"""
    load: cp.Variable
    """storage loading"""
    # logical variables
    bin_load: cp.Variable
    """logical variable indicating if storage is loading"""
    bin_gen: cp.Variable
    """logical variable indicating if storage is generating"""

    @classmethod
    def create(cls, indices: Indices) -> Self:
        tt = len(indices.t_idx.vals)
        return cls(
            soc=cp.Variable(tt, name='soc', nonneg=True),
            gen=cp.Variable(tt, name='gen', nonneg=True),
            load=cp.Variable(tt, name='load', nonneg=True),
            bin_load=cp.Variable(tt, name='bin_load', boolean=True),
            bin_gen=cp.Variable(tt, name='bin_gen', boolean=True),
        )
