from pydantic import BaseModel

from src.preprocess.model.market import MarketData
from src.preprocess.model.params import Params
from src.preprocess.model.storage import StorageParams


class InputData(BaseModel):
    """Raw input data"""
    market_data: MarketData
    """energy market data"""
    storage_params: StorageParams
    """energy storage characteristics"""
    params: Params
    """optimization parameters"""
