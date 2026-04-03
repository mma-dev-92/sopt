from pydantic import BaseModel, Field


class MinMaxBounds(BaseModel):
    min: float
    max: float


class EconomicStorageParams(BaseModel):
    wacc: float = Field(
        ..., description="Weighted Average Cost of Capital (e.g., 0.10 for 10%). Used for discounting future cash flows."
    )
    capex: float = Field(
        ..., description="Total Capital Expenditure. Typically provided in PLN/MWh of nameplate capacity."
    )
    opex: float = Field(
        ..., description="Annual Operational Expenditure. Typically provided in PLN/MWh/year."
    )


class TechnicalStorageParams(BaseModel):
    capacity: float = Field(
        ..., description="Nominal nameplate energy capacity [MWh]. The 'size' of the battery tank."
    )
    power: float = Field(
        ..., description="Maximum instantaneous power output/input [MW]. Defines the C-rate."
    )
    charge_efficiency: float = Field(
        ..., description="Efficiency of energy conversion during charging [0-1]. Internal losses and AC/DC conversion."
    )
    discharge_efficiency: float = Field(
        ..., description="Efficiency of energy conversion during discharging [0-1]."
    )
    soc_limits: MinMaxBounds = Field(
        ..., description="Operational boundaries for State of Charge to prevent deep discharge damage."
    )


class CapacityDegradationParams(BaseModel):
    max_capacity_loss: float = Field(
        ..., description="The End-of-Life (EOL) threshold as a fraction of initial capacity (e.g., 0.2 for 20% loss)."
    )
    lifetime_years: int = Field(
        ..., description="The manufacturer's warrantied calendar life under nominal conditions."
    )
    n_cycles: int = Field(
        ..., description="The manufacturer's rated cycle life at the reference Depth of Discharge (DoD)."
    )
    time_decay_duration_years: float = Field(
        ..., description="The theoretical time it would take for the battery to hit EOL purely through calendar aging at reference T/SOC."
    )

class CapacityDegradationModelParams(BaseModel):
    """
    Parameters for the semi-empirical battery degradation_model model, governing
    both cyclic (mechanical/kinetic) and calendar (static/potential) aging.
    """
    reference_dod: float = Field(
        ..., description="The depth of discharge (δ) used as the basis for the power law "
                         "lifetime calculation (e.g., 0.8 for 80% DoD swings)."
    )
    soc_sensitivity: float = Field(
        ..., description="The coefficient (α) in the exponential stress function for "
                         "calendar aging. Quantifies how high SOC accelerates chemical decay."
    )
    temperature_offset: float = Field(
        ..., description="The constant thermal gradient (ΔT) in Kelvin between the "
                         "ambient environment and internal cells."
    )
    activation_energy: float = Field(
        ..., description="Gibbs activation energy (Ea) in J/mol. Defines temperature sensitivity."
    )
    reference_temperature: float = Field(
        298.15, description="Standard reference temperature in Kelvin (25°C)."
    )


class StorageStaticParams(BaseModel):
    """
    High-level container for immutable storage characteristics.
    """
    technical: TechnicalStorageParams
    economics: EconomicStorageParams
    degradation: CapacityDegradationParams
    deg_model: CapacityDegradationModelParams


class StorageStateParams(BaseModel):
    """
    The 'Digital Twin' state. These values evolve as the simulation progresses.
    """
    soc: float = Field(
        ..., description="Current State of Charge [0.0 - 1.0]. Represents the starting point for optimization."
    )
    capacity_loss: float = Field(
        ..., description="Current cumulative health degradation_model [0.0 - max_capacity_loss]."
    )
