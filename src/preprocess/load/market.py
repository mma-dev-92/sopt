import pandas as pd
from pathlib import Path

from src.preprocess.model.market import MarketData


def load_price_data_from_entsoe(path: Path) -> pd.DataFrame:
    """
    Load DA market price data from entsoe (from CSV file)
    :param path: path to the file
    :return: DataFrame containing market price data
    """

    data = pd.read_csv(path)

    price_col = 'Day-ahead Price (EUR/MWh)'
    data[["start_str", "end_str"]] = data["MTU (UTC)"].str.split(" - ", expand=True)
    data["timestamp"] = pd.to_datetime(data["start_str"], format="%d/%m/%Y %H:%M:%S")
    data = data.rename(columns={price_col: 'price'})
    data = data[['timestamp', 'price']].set_index('timestamp')
    return data


def get_data_resolution(df: pd.DataFrame) -> float:
    # get timestamp index
    t_index = df.index.get_level_values('timestamp')

    # compute interval length
    intervals = t_index.diff()[1:]
    if not intervals.nunique() == 1:
        raise ValueError("all rows in price data must have the same resolution")

    return intervals.unique().values.squeeze() / pd.Timedelta(hours=1)


def load_market_data(path: Path) -> MarketData:
    price_df = load_price_data_from_entsoe(path)
    resolution = get_data_resolution(price_df)
    return MarketData(prices=price_df, resolution=resolution)
