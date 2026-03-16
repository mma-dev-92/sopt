import enum

import pandera.pandas as pa
from pandera.typing import Series
import pandas as pd


class MarketType(str, enum.Enum):
    DAM = "DAM"
    ID1 = "ID1"
    ID2 = "ID2"
    ID3 = "ID3"


class Currency(str, enum.Enum):
    PLN = "PLN"
    EUR = "EUR"


class EnergyUnit(str, enum.Enum):
    MWH = "MWH"
    KWH = "KWH"


class PowerUnit(str, enum.Enum):
    MW = "MW"


class DataType(str, enum.Enum):
    FORECAST = "FORECAST"
    OBSERVATION = "OBSERVATION"


class DataSource(str, enum.Enum):
    ENTSOE = "ENTSOE"
    PSE = "PSE"
    OPEN_METEO = "OPEN_METEO"
    OMIE = "OMIE"


class BronzeBaseSchema(pa.DataFrameModel):
    """
    Base schema for bronze-layer datasets.

    Time conventions:
    - All timestamps MUST be timezone-aware and in UTC
    - valid_time = start of interval the record refers to
    - issue_time = when the provider published the data
    - snapshot = when we ingested the data
    """

    snapshot: Series[pd.Timestamp] = pa.Field(nullable=False)
    issue_time: Series[pd.Timestamp] = pa.Field(nullable=False)
    valid_time: Series[pd.Timestamp] = pa.Field(nullable=False)

    data_source: Series[str] = pa.Field(isin=[e.value for e in DataSource])
    data_type: Series[str] = pa.Field(nullable=True, isin=[e.value for e in DataType])

    class Config:
        strict = True
        coerce = True
    
    @pa.dataframe_check
    def time_order_consistent(cls, df: pd.DataFrame) -> bool:
        """
        Ensures timestamps follow logical order:
        issue_time <= snapshot
        """
        return (df["issue_time"] <= df["snapshot"]).all()


class MarketPriceSchema(BronzeBaseSchema):

    market: Series[str] = pa.Field(nullable=False, isin=[e.value for e in MarketType])
    price: Series[float] = pa.Field(nullable=True)

    currency: Series[str] = pa.Field(nullable=False, isin=[e.value for e in Currency])
    energy_unit: Series[str] = pa.Field(nullable=False, isin=[e.value for e in EnergyUnit])
    exchange_rate_to_pln: Series[float] = pa.Field(nullable=False)


class WeatherSchema(BronzeBaseSchema):

    latitude: Series[float] = pa.Field(nullable=False)
    longitude: Series[float] = pa.Field(nullable=False)

    irradiance: Series[float] = pa.Field(nullable=True)  # solar energy at the surface level [W/m²]
    temperature: Series[float] = pa.Field(nullable=True)  # °C

    cloud_cover: Series[float] = pa.Field(nullable=True)  # %
    wind_speed_10m: Series[float] = pa.Field(nullable=True)  # m/s


class GridStateSchema(BronzeBaseSchema):

    system_load: Series[float] = pa.Field(nullable=True)
    solar_generation: Series[float] = pa.Field(nullable=True)
    wind_offshore_generation: Series[float] = pa.Field(nullable=True)
    wind_onshore_generation: Series[float] = pa.Field(nullable=True)

    power_unit: Series[str] = pa.Field(nullable=False, isin=[e.value for e in PowerUnit])
