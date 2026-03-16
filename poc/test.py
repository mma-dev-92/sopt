import pandas as pd

from poc.data.bronze.dataset import MarketPriceDataset, GridStateDataset, WeatherDataset
from poc.data.bronze.schema import MarketPriceSchema, GridStateSchema, WeatherSchema
from poc.data.bronze.metadata import BronzeBaseDatasetMetadata

start = pd.Timestamp("2024-10-01", tz="UTC")
end   = pd.Timestamp("2024-10-02", tz="UTC")

market_price_metadata = BronzeBaseDatasetMetadata.from_yaml(
    "/home/maciek/Projects/sopt/poc/assets/datasets/market_prices.yaml",
    schema=MarketPriceSchema
)

grid_state_metadata = BronzeBaseDatasetMetadata.from_yaml(
    "/home/maciek/Projects/sopt/poc/assets/datasets/grid_state.yaml",
    schema=GridStateSchema,
)

weather_metadata = BronzeBaseDatasetMetadata.from_yaml(
    "/home/maciek/Projects/sopt/poc/assets/datasets/weather.yaml",
    schema=WeatherSchema,
)

market_price_dataset = MarketPriceDataset(
    metadata=market_price_metadata,
    start=start,
    end=end,
)

grid_state_dataset = GridStateDataset(
    metadata=grid_state_metadata,
    api_key="b25938c6-f2d3-4eda-aef2-8d71580afdcc",
    start=start,
    end=end,
    country_code="ES",
)

market_df = market_price_dataset.load()
grid_state_df = grid_state_dataset.load()
