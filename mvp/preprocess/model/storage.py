from pydantic import BaseModel


class StorageParams(BaseModel):
    # technical parameters
    capacity: float
    """storage capacity"""
    power: float
    """storage power"""
    load_efficiency: float
    """load efficiency"""
    gen_efficiency: float
    """generation efficiency"""
    dod: tuple[float, float]
    """min, max depth of discharge"""
    # degradation rate (to do appropriate research - not current phase)
    ...
    # financing parameters
    wacc: float
    """weighted average cost of capital for the investment"""
    capex: float
    """investment cost (assumed PLN/MWh)"""
    # for now, let's assume opex = epsilon * capex (yearly)
    opex: float
    """operational costs (assumed PLN/MWh)"""
