import pandas as pd
from pathlib import Path

from mvp.preprocess.model.market import MarketData


def load_price_data_from_entsoe(path: Path) -> pd.DataFrame:
    """
    Load DA market price data from entsoe (from CSV file)
    :param path: path to the file
    :return: DataFrame containing market price data
    """

    data = pd.read_csv(path)

    price_col = 'Day-ahead Price (EUR/MWh)'
    data[["start_str", "end_str"]] = data["MTU (UTC)"].str.split(" - ", expand=True)
    data["start"] = pd.to_datetime(data["start_str"], format="%d/%m/%Y %H:%M:%S")
    data["end"] = pd.to_datetime(data["end_str"], format="%d/%m/%Y %H:%M:%S")
    data = data.rename(columns={price_col: 'price'})
    data = data[['start', 'end', 'price']].set_index(['start', 'end'])
    return data


def get_data_resolution(df: pd.DataFrame) -> float:
    # Ensure datetime dtype
    df['start'] = pd.to_datetime(df['start'])
    df['end'] = pd.to_datetime(df['end'])

    # Compute interval length
    interval = df['end'] - df['start']
    if not interval.nunique() == 1:
        raise ValueError("all rows in price data must have the same resolution")

    return interval / pd.Timedelta(hours=1)


def load(path: Path) -> MarketData:
    price_df = load_price_data_from_entsoe(path)
    resolution = get_data_resolution(price_df)
    return MarketData(prices=price_df, resolution=resolution)
