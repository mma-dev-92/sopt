import dataclasses
import cvxpy as cp

from typing import Self

import numpy as np

from src.opt.indices import Indices
from src.preprocess.model import StorageStaticParams, InputData


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
    dt: float
    """time resolution"""

    @classmethod
    def create(cls, input_data: InputData) -> Self:
        ssp = input_data.storage_static_params
        state = input_data.storage_state_params

        return cls(
            charge_eta=ssp.technical.charge_efficiency,
            discharge_eta=ssp.technical.discharge_efficiency,
            soc_rel_lb=ssp.technical.soc_limits.min,
            soc_rel_ub=ssp.technical.soc_limits.max,
            nom_p=ssp.technical.power,
            dt=input_data.market_data.resolution,
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
    def create(cls, input_data: InputData) -> Self:
        tt = int(24 / input_data.market_data.resolution)
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
    def create(cls, input_data: InputData, indices: Indices) -> Self:
        return cls(
            static=StaticOptParams.create(input_data),
            dynamic=DynamicOptParams.create(input_data),
        )
