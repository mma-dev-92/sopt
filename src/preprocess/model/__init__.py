import numpy as np
import pandas as pd
from pydantic import BaseModel

from src.preprocess.model.market import MarketData
from src.preprocess.model.params import Params
from src.preprocess.model.storage import StorageStaticParams, StorageStateParams


class InputData(BaseModel):
    """Raw input data"""
    market_data: MarketData
    """energy market data"""
    storage_static_params: StorageStaticParams
    """energy storage static parameters"""
    storage_state_params: StorageStateParams
    """energy storage state"""
    params: Params
    """optimization parameters"""
