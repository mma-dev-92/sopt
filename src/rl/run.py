import numpy as np
import pandas as pd

from config import ROOT
from poc.data.bronze.dataset import EntsoeMarketPriceDataset
from poc.data.bronze.metadata import BronzeBaseDatasetMetadata
from poc.data.bronze.schema import MarketPriceSchema
from src.preprocess.load import load_storage_static_params
from src.rl.environment import BatteryEnvironment
from src.rl.spaces import Action

# start = pd.Timestamp("2016-01-02", tz="UTC")
# end   = pd.Timestamp("2026-01-01", tz="UTC")
#
# # load the meta-data (from *.yaml file; pydantic-validated)
# market_price_metadata = BronzeBaseDatasetMetadata.from_yaml(
#     str(ROOT / "poc/assets/datasets/entsoe_dam_prices.yaml"),
#     schema=MarketPriceSchema
# )
#
# market_price_dataset = EntsoeMarketPriceDataset(
#     metadata=market_price_metadata,
#     start=start,
#     end=end,
#     api_key="b25938c6-f2d3-4eda-aef2-8d71580afdcc",
# )

dam_prices = pd.read_parquet(ROOT / 'assets' / 'input' / 'dam_prices.parquet')
ssp = load_storage_static_params(ROOT / 'assets' / 'input' / 'storage_static_params.yaml')
max_days = int(len(dam_prices) / 24)

env = BatteryEnvironment(
    storage_static_params=ssp,
    energy_prices=dam_prices['price'],
    dt=1.0,
    n_hours=24,
    max_days=max_days,
)

action = Action(lambda_param=np.ones(24))
env.reset()
transition_result = env.step(action)
