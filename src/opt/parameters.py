import dataclasses
from dataclasses import fields

import cvxpy as cp

from typing import Self

from src.opt.indices import TimeIndex
from src.preprocess.model import InputData, StorageStaticParams


@dataclasses.dataclass
class StaticOptParams:

    charge_eta: float
    """generation efficiency"""
    discharge_eta: float
    """load efficiency"""
    soc_rel_lb: float
    """min soc/cap"""
    soc_rel_ub: float
    """max soc/cap"""
    nom_p: float
    """nominal power (for charging and discharging)"""

    @classmethod
    def create(cls, storage_static_params: StorageStaticParams) -> Self:

        return cls(
            charge_eta=storage_static_params.technical.charge_efficiency,
            discharge_eta=storage_static_params.technical.discharge_efficiency,
            soc_rel_lb=storage_static_params.technical.soc_limits.min,
            soc_rel_ub=storage_static_params.technical.soc_limits.max,
            nom_p=storage_static_params.technical.power,
        )


@dataclasses.dataclass
class DynamicOptParams:
    """
    Dynamic parameters - to avoid cvxpy model compilation before each run.
    """

    lambda_penalty: cp.Parameter
    """Hyper parameter for adjusting the objective function penalization (per timestep)"""
    init_soc: cp.Parameter
    """Initial state of charge - boundary condition"""
    cap: cp.Parameter
    """Capacity of the storage (will decrease in time because of the capacity degradation)"""
    energy_price: cp.Parameter
    """Energy price for a given day"""

    @classmethod
    def create(cls, t_idx: TimeIndex) -> Self:
        tt = len(t_idx.vals)
        return cls(
            lambda_penalty=cp.Parameter(shape=(tt, ), name="lambda"),
            energy_price=cp.Parameter(shape=(tt, ), name="energy_price"),
            init_soc=cp.Parameter(nonneg=True, name="init_soc"),
            cap=cp.Parameter(nonneg=True, name="cap"),
        )


@dataclasses.dataclass
class Parameters:
    static: StaticOptParams
    dynamic: DynamicOptParams

    @classmethod
    def create(cls, storage_static_params: StorageStaticParams, t_idx: TimeIndex) -> Self:
        return cls(
            static=StaticOptParams.create(storage_static_params),
            dynamic=DynamicOptParams.create(t_idx),
        )
