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
    rev: cp.Variable
    """revenue for each timestamp"""
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
            rev=cp.Variable(tt, name='rev'),
            bin_load=cp.Variable(tt, name='bin_load', boolean=True),
            bin_gen=cp.Variable(tt, name='bin_gen', boolean=True),
        )

    @property
    def decision_variables(self) -> list[cp.Variable]:
        return [self.gen, self.load, self.soc, self.rev]

    @property
    def logical_variables(self) -> list[cp.Variable]:
        return [self.bin_load, self.bin_gen]

    @property
    def all_variables(self) -> list[cp.Variable]:
        return self.decision_variables + self.logical_variables
