import cvxpy as cp
import dataclasses

from typing import Self

from src.opt.indices import TimeIndex


@dataclasses.dataclass
class Variables:
    # decision variables
    soc: cp.Variable
    """storage state of charge"""
    discharge: cp.Variable
    """storage generation"""
    charge: cp.Variable
    """storage loading"""
    rev: cp.Variable
    """revenue for each timestamp"""
    # logical variables
    bin_load: cp.Variable
    """logical variable indicating if storage is loading"""
    bin_gen: cp.Variable
    """logical variable indicating if storage is generating"""

    @classmethod
    def create(cls, t_idx: TimeIndex) -> Self:
        tt = len(t_idx)
        return cls(
            soc=cp.Variable(tt, name='soc', nonneg=True),
            discharge=cp.Variable(tt, name='discharge', nonneg=True),
            charge=cp.Variable(tt, name='charge', nonneg=True),
            rev=cp.Variable(tt, name='rev'),
            bin_load=cp.Variable(tt, name='bin_load', boolean=True),
            bin_gen=cp.Variable(tt, name='bin_gen', boolean=True),
        )

    @property
    def decision_variables(self) -> list[cp.Variable]:
        return [self.discharge, self.charge, self.soc, self.rev]

    @property
    def logical_variables(self) -> list[cp.Variable]:
        return [self.bin_load, self.bin_gen]

    @property
    def all_variables(self) -> list[cp.Variable]:
        return self.decision_variables + self.logical_variables
