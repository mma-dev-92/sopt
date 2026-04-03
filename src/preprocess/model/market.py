from pandas import Series
from pydantic import BaseModel, Field


class MarketData(BaseModel):
    prices: Series
    resolution: float = Field(gt=0, le=1.0)

    class Config:
        arbitrary_types_allowed = True  # required for pandas

