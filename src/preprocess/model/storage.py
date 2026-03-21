from pydantic import BaseModel


class MinMaxBounds(BaseModel):
    min: float
    max: float


class EconomicStorageParams(BaseModel):
    wacc: float
    """weighted average cost of capital for the investment"""
    capex: float
    """investment cost (assumed PLN/MWh)"""
    opex: float
    """operational costs (assumed PLN/MWh)"""


class TechnicalStorageParams(BaseModel):
    capacity: float
    """storage nominal (initial) capacity"""
    power: float
    """storage power"""
    charge_efficiency: float
    """charge efficiency"""
    discharge_efficiency: float
    """discharge efficiency"""
    soc_limits: MinMaxBounds
    """min & max values of the depth of discharge"""


class CapacityDegradationParams(BaseModel):
    max_capacity_loss: float
    """
    at which point storage dies 
    cap_eol = cap_0 * max_capacity_loss
    """
    dod_segment_fraction: list[float]
    """dod cycle depth capacity degradation segmentation"""
    lifetime_years: int
    """life time of a storage"""
    n_cycles: int
    """number of cycles to be performed during storage life time"""
    time_decay_duration_years: float
    """time (years) of time (calendar) decay from cap_0 to cap_eol"""
    dod_avg: float
    """assumed average depth of a cycle"""


class StorageStaticParams(BaseModel):
    """
    Storage parameters that are constant in time
    """
    technical: TechnicalStorageParams
    economics: EconomicStorageParams
    degradation: CapacityDegradationParams



class StorageStateParams(BaseModel):
    """
    Storage state parameters (evolve in time)
    """
    soc: float
    """current state of charge (initial state of charge to opt problem)"""
    capacity_loss: float
    """current level of capacity degradation"""
