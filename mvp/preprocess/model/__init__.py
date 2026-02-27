from pydantic import BaseModel

from mvp.preprocess.model.market import MarketData
from mvp.preprocess.model.storage import StorageParams


class InputData(BaseModel):
    """Raw input data"""
    market_data: MarketData
    """energy market data"""
    storage_params: StorageParams
    """energy storage characteristics"""
