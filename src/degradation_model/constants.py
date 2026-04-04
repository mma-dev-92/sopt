import enum


class Constants(float, enum.Enum):
    R = 8.314
    """Universal gas constant [J/mol]"""
    CYCLE_SEVERITY_EXPONENT = 1.3
    """Cycle severity exponent in cycle degradation_model model"""
    opt_temperature = 298.15
    """25 degree celsius temperature in Kelvin - optimal temperature for battery to operate"""
