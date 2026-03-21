import dataclasses
import numpy as np

from typing import Self

from src.opt.indices import Indices
from src.preprocess.model import StorageStaticParams, InputData

_DAYS_PER_YEAR = 365
_HOURS_PER_DAY = 24


@dataclasses.dataclass
class StorageOptParams:

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
    nom_cap: float
    """nominal capacity (given by producer, dod / degradation not in"""
    cap_loss: float
    """fraction of nominal capacity degradated so far"""
    max_cap_loss: float
    """maximal capacity loss, so the degradation is at most cap_t >= nom_cap * max_cap_loss"""
    init_soc: float
    """initial state of charge"""
    alpha: float
    """cycle depth degradation coefficient"""
    beta: float
    """aging degradation coefficient"""
    w: np.ndarray
    """cycle depth penalization weights"""
    s: np.ndarray
    """dod segment fractions"""

    @classmethod
    def create(cls, input_data: InputData) -> Self:
        ssp = input_data.storage_static_params
        state = input_data.storage_state_params
        dcap_calendar, dcap_cycle = cls.compute_dcap_decomposition(input_data)

        return cls(
            charge_eta=ssp.technical.charge_efficiency,
            discharge_eta=ssp.technical.discharge_efficiency,
            soc_rel_lb=ssp.technical.soc_limits.min,
            soc_rel_ub=ssp.technical.soc_limits.max,
            nom_p=ssp.technical.power,
            nom_cap=ssp.technical.capacity,
            cap_loss=state.capacity_loss,
            max_cap_loss=ssp.degradation.max_capacity_loss,
            init_soc=state.soc,
            s=np.array(ssp.degradation.dod_segment_fraction),
            alpha=cls.compute_alpha(input_data, dcap_cycle),
            beta=cls.compute_beta(input_data, dcap_calendar),
            w=cls.compute_w(input_data),
        )

    @staticmethod
    def compute_dcap_decomposition(input_data: InputData) -> tuple[float, float]:
        """compute dcap total = dcap calendar + dcap cycle decomposition"""
        n_years = input_data.storage_static_params.degradation.lifetime_years
        calendar_degradation = input_data.storage_static_params.degradation.calendar_fade_per_year
        nom_cap = input_data.storage_static_params.technical.capacity
        max_cap_loss = input_data.storage_static_params.degradation.max_capacity_loss

        dcap_calendar = nom_cap * n_years * calendar_degradation
        dcap_total = nom_cap * (1 - max_cap_loss)

        return dcap_calendar, dcap_total - dcap_calendar


    @staticmethod
    def compute_alpha(input_data: InputData, dcap_cycle: float) -> float:
        """
        coefficient that measures capacity degradation per storage energy throughput
        """
        n_cycles = input_data.storage_static_params.degradation.n_cycles
        cap = input_data.storage_static_params.technical.capacity
        max_cap_loss = input_data.storage_static_params.degradation.max_capacity_loss
        t_eol = n_cycles * cap * (1 - max_cap_loss) / 2

        return dcap_cycle / t_eol

    @staticmethod
    def compute_beta(input_data: InputData, dcap_calendar: float) -> float:
        """beta describes capacity degradation per timestep, does not depend on n_cycles, just time"""
        dt = input_data.market_data.resolution
        n_years = input_data.storage_static_params.degradation.lifetime_years
        n_steps = int(n_years * _DAYS_PER_YEAR * _HOURS_PER_DAY / dt)

        return dcap_calendar / n_steps

    @staticmethod
    def compute_w(input_data: InputData) -> np.ndarray:
        """capacity cycle degradation weights"""
        def _F(x: float) -> float:
            """_F(DoD) is proportional to the capacity degradation severeness"""
            return x ** 1.2

        s = input_data.storage_static_params.degradation.dod_segment_fraction
        w = np.zeros_like(s)

        s_cum_sum = 0.0
        for i in range(len(s)):
            w[i] = _F(s_cum_sum + s[i]) - _F(s_cum_sum)
            s_cum_sum += s[i]

        return w


@dataclasses.dataclass
class Parameters:
    prices: np.ndarray
    dt: float
    storage_opt_params: StorageOptParams

    @classmethod
    def create(cls, input_data: InputData, indices: Indices) -> Self:
        return cls(
            prices=input_data.market_data.prices.loc[indices.t_idx.vals, :].values,
            dt=input_data.market_data.resolution,
            storage_opt_params=StorageOptParams.create(input_data),
        )
