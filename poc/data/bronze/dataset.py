import datetime
import os
from pathlib import Path
from abc import ABC, abstractmethod
from datetime import timedelta
from io import StringIO

import pandas as pd
import requests
from entsoe import EntsoePandasClient

from poc.data.bronze.schema import (
    MarketPriceSchema,
    MarketType,
    Currency,
    EnergyUnit,
    DataSource, DataType, PowerUnit,
)

from poc.data.bronze.metadata import BronzeBaseDatasetMetadata, TimeResolution

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class BronzeDataset(ABC):

    _time_cols = ["snapshot", "issue_time", "valid_time"]

    def __init__(
            self,
            metadata: BronzeBaseDatasetMetadata,
            start: pd.Timestamp,
            end: pd.Timestamp,
    ):
        self.metadata = metadata
        self.schema = metadata.data_schema

        self.start = start
        self.end = end

    @abstractmethod
    def fetch(self) -> pd.DataFrame:
        """
        Fetch data from external source.
        Must be implemented by subclasses.
        """
        raise NotImplementedError

    def validate(self, df):

        self.ensure_utc_timestamps(df)
        df = self.schema.validate(df)
        df = self.restore_utc_timestamps(df)

        return df

    def load(self):

        df = self.fetch()
        df = self.validate(df)

        return df

    @staticmethod
    def write(df, path: Path):

        path.parent.mkdir(parents=True, exist_ok=True)

        df.to_parquet(path, index=False)

    @staticmethod
    def ensure_utc_timestamps(df):
        for c in BronzeDataset._time_cols:
            if df[c].dt.tz != datetime.timezone.utc:
                raise ValueError(f"{c} must be UTC")

        return df

    @staticmethod
    def restore_utc_timestamps(df):
        for c in BronzeDataset._time_cols:
            if df[c].dt.tz is None:
                df[c] = df[c].dt.tz_localize("UTC")
        return df


class EntsoeMarketPriceDataset(BronzeDataset):
    """
    Independent ENTSO-E fetcher.
    Dynamically adjusts resolution and filters incomplete days.
    """

    def __init__(
            self,
            metadata: BronzeBaseDatasetMetadata,
            start: pd.Timestamp,
            end: pd.Timestamp,
            country_code: str = "PL",
            api_key: str | None = None,
            filter_incomplete_days: bool = True,
    ):
        super().__init__(metadata, start, end)
        self.country_code = country_code
        self.filter_incomplete_days = filter_incomplete_days
        self.api_key = api_key or os.getenv("ENTSOE_API_KEY")

        if not self.api_key:
            raise ValueError("ENTSO-E API Key is missing.")

        self.client = EntsoePandasClient(api_key=self.api_key)

    def fetch(self) -> pd.DataFrame:
        """
        Fetches, cleans, and transforms data based on metadata resolution.
        """
        snapshot = pd.Timestamp.now(tz="UTC")

        try:
            series = self.client.query_day_ahead_prices(
                self.country_code,
                start=self.start,
                end=self.end
            )
        except Exception as e:
            logger.error(f"ENTSO-E fetch failed: {e}")
            raise RuntimeError(f"Fetch failure: {e}")

        if series.empty:
            return pd.DataFrame(columns=MarketPriceSchema.to_schema().columns.keys())

        # 1. Standardize basic structure
        df = series.to_frame(name="price").reset_index()
        df.rename(columns={"index": "valid_time"}, inplace=True)
        df["valid_time"] = pd.to_datetime(df["valid_time"]).dt.tz_convert("UTC")

        # 2. Add Required Columns
        df["snapshot"] = snapshot
        df["data_source"] = DataSource.ENTSOE.value
        df["data_type"] = DataType.OBSERVATION.value
        df["market"] = MarketType.DAM.value
        df["currency"] = Currency.EUR.value
        df["energy_unit"] = EnergyUnit.MWH.value
        df["exchange_rate_to_pln"] = 1.0

        # Calculate issue_time (12:00 CET/CEST day before)
        df["issue_time"] = (
                df["valid_time"].dt.tz_convert("Europe/Warsaw")
                .dt.normalize() - pd.Timedelta(days=1)
                + pd.Timedelta(hours=12)
        ).dt.tz_convert("UTC")

        # 3. Resolution Logic
        # Only upsample if metadata explicitly asks for 15min
        if self.metadata.valid_time_resolution == TimeResolution.MIN15:
            df = self._upsample_to_15min(df)
            logger.info("Upsampled ENTSO-E data to 15min resolution.")
        else:
            logger.info("Preserving original hourly ENTSO-E resolution.")

        # 4. Clean incomplete days (Remove any day with a NaN price)
        if self.filter_incomplete_days:
            df = self._filter_incomplete_days(df)

        return df[list(MarketPriceSchema.to_schema().columns.keys())]

    def _upsample_to_15min(self, df: pd.DataFrame) -> pd.DataFrame:
        """Forward-fills hourly prices into 15-minute intervals."""
        return (
            df.set_index("valid_time")
            .resample("15min")
            .ffill()
            .reset_index()
        )

    def _filter_incomplete_days(self, df: pd.DataFrame) -> pd.DataFrame:
        """Removes all intervals for a day if any interval in that day is NaN."""
        # Group by the date part of the UTC valid_time
        temp_date = df['valid_time'].dt.date
        is_nan_day = df['price'].isna().groupby(temp_date).transform('any')

        df_clean = df[~is_nan_day].copy()

        dropped = temp_date.nunique() - df_clean['valid_time'].dt.date.nunique()
        if dropped > 0:
            logger.warning(f"Dropped {dropped} incomplete days from dataset.")

        return df_clean
class MarketPriceDataset(BronzeDataset):

    BASE_URL = "https://www.omie.es/en/file-download"

    INTRADAY_SESSION_MAP = {
        "01": MarketType.ID1,
        "02": MarketType.ID2,
        "03": MarketType.ID3,
    }

    def __init__(
        self,
        metadata,
        start: pd.Timestamp,
        end: pd.Timestamp,
    ):
        super().__init__(metadata, start, end)


    def fetch(self) -> pd.DataFrame:

        intraday = self.fetch_intraday_prices()
        dam = self.fetch_day_ahead_prices()

        dfs = [df for df in [intraday, dam] if df is not None]

        if not dfs:
            raise RuntimeError("No market data fetched")

        return pd.concat(dfs, ignore_index=True)


    def fetch_intraday_prices(self) -> pd.DataFrame | None:

        snapshot = pd.Timestamp.now(tz="UTC")

        dfs = []

        date = self.start.normalize()

        while date <= self.end:

            for session, market in self.INTRADAY_SESSION_MAP.items():

                fname = f"marginalpibc_{date:%Y%m%d}{session}.1"

                df = self._download_file(
                    f_name=fname,
                    parent="marginalpibc",
                    market=market,
                    snapshot=snapshot,
                )

                if df is not None:
                    dfs.append(df)

            date += timedelta(days=1)

        if not dfs:
            return None

        return pd.concat(dfs, ignore_index=True)


    def fetch_day_ahead_prices(self) -> pd.DataFrame | None:

        snapshot = pd.Timestamp.now(tz="UTC")

        dfs = []

        date = self.start.normalize()

        while date <= self.end:

            f_name = f"marginalpdbc_{date:%Y%m%d}.1"

            df = self._download_file(
                f_name=f_name,
                parent="marginalpdbc",
                market=MarketType.DAM,
                snapshot=snapshot,
            )

            if df is not None:
                dfs.append(df)

            date += timedelta(days=1)

        if not dfs:
            return None

        return pd.concat(dfs, ignore_index=True)


    def _download_file(
        self,
        f_name: str,
        parent: str,
        market: MarketType,
        snapshot: pd.Timestamp,
    ) -> pd.DataFrame | None:

        params = {
            "parents": parent,
            "filename": f_name,
        }

        try:

            r = requests.get(self.BASE_URL, params=params, timeout=30)

            if r.status_code != 200:
                logger.warning(f"missing {f_name}")
                return None

            df = pd.read_csv(
                StringIO(r.text),
                sep=";",
                header=None,
                skiprows=1,
                comment="*",
                engine="python",
            )

            df = df.iloc[:, :6]

            df.columns = [
                "year",
                "month",
                "day",
                "period",
                "price_es",
                "price_pt",
            ]

            df["price_es"] = pd.to_numeric(df["price_es"], errors="coerce")

            df["valid_time"] = (
                pd.to_datetime(df[["year", "month", "day"]])
                + pd.to_timedelta((df["period"] - 1) * 15, unit="min")
            )

            df["valid_time"] = (
                df["valid_time"]
                .dt.tz_localize("Europe/Madrid")
                .dt.tz_convert("UTC")
            )

            df["snapshot"] = snapshot

            # ---------- issue time logic ----------
            delivery_date = pd.to_datetime(df[["year", "month", "day"]]).iloc[0]
            if market == MarketType.DAM:
                issue_time = (
                        (delivery_date - pd.Timedelta(days=1))
                        + pd.Timedelta(hours=12)
                )
            elif market == MarketType.ID1:
                issue_time = (
                        (delivery_date - pd.Timedelta(days=1))
                        + pd.Timedelta(hours=15)
                )
            elif market == MarketType.ID2:
                issue_time = (
                        (delivery_date - pd.Timedelta(days=1))
                        + pd.Timedelta(hours=22)
                )
            elif market == MarketType.ID3:
                issue_time = delivery_date + pd.Timedelta(hours=10)
            else:
                raise ValueError(f"unrecognized market type {market}")

            issue_time = (
                issue_time
                .tz_localize("Europe/Madrid")
                .tz_convert("UTC")
            )

            df["snapshot"] = snapshot
            df["issue_time"] = issue_time

            df["data_source"] = DataSource.OMIE.value
            df["data_type"] = DataType.OBSERVATION.value

            df["market"] = market.value
            df["price"] = df["price_es"]

            df["currency"] = Currency.EUR.value
            df["energy_unit"] = EnergyUnit.MWH.value
            df["exchange_rate_to_pln"] = 1.0

            return df[
                [
                    "snapshot",
                    "issue_time",
                    "valid_time",
                    "data_source",
                    "data_type",
                    "market",
                    "price",
                    "currency",
                    "energy_unit",
                    "exchange_rate_to_pln",
                ]
            ]

        except Exception as e:

            logger.error(f"failed parsing {fname}: {e}")
            return None


class WeatherDataset(BronzeDataset):

    def __init__(
            self,
            metadata: BronzeBaseDatasetMetadata,
            start: pd.Timestamp,
            end: pd.Timestamp,
            latitude: float,
            longitude: float
    ) -> None:
        super().__init__(metadata, start, end)

        self.latitude = latitude
        self.longitude = longitude

    def fetch(self):
        now = pd.Timestamp.now(tz="UTC")

        archive_url = "https://archive-api.open-meteo.com/v1/archive"

        params = {
            "latitude": self.latitude,
            "longitude": self.longitude,
            "start_date": self.start.date(),
            "end_date": self.end.date(),
            "hourly": [
                "temperature_2m",
                "cloudcover",
                "windspeed_10m",
                "shortwave_radiation",
            ],
            "timezone": "UTC",
        }

        r = requests.get(archive_url, params=params)
        r.raise_for_status()

        data = r.json()["hourly"]

        df = pd.DataFrame({
            "valid_time": pd.to_datetime(data["time"], utc=True),
            "temperature": data["temperature_2m"],
            "cloud_cover": data["cloudcover"],
            "wind_speed_10m": data["windspeed_10m"],
            "irradiance": data["shortwave_radiation"],
        })

        df["snapshot"] = now
        df["issue_time"] = now

        df["data_source"] = DataSource.OPEN_METEO.value
        df["data_type"] = DataType.OBSERVATION.value

        df["latitude"] = self.latitude
        df["longitude"] = self.longitude

        df = df[
            [
                "snapshot",
                "issue_time",
                "valid_time",
                "data_source",
                "data_type",
                "latitude",
                "longitude",
                "irradiance",
                "temperature",
                "cloud_cover",
                "wind_speed_10m",
            ]
        ]

        return df

class GridStateDataset(BronzeDataset):

    def __init__(
            self,
            metadata: BronzeBaseDatasetMetadata,
            start: pd.Timestamp,
            end: pd.Timestamp,
            api_key: str,
            country_code: str ="PL"
    ) -> None:
        super().__init__(metadata, start, end)

        self.client = EntsoePandasClient(api_key=api_key)
        self.country_code = country_code

    def fetch(self):
        now = pd.Timestamp.now(tz="UTC")

        load = self.client.query_load(
            self.country_code,
            start=self.start,
            end=self.end,
        )

        solar = self.client.query_generation(
            self.country_code,
            start=self.start,
            end=self.end,
            psr_type="B16",
        )

        wind_offshore = self.client.query_generation(
            self.country_code,
            start=self.start,
            end=self.end,
            psr_type="B18",
        )

        wind_onshore = self.client.query_generation(
            self.country_code,
            start=self.start,
            end=self.end,
            psr_type="B19",
        )

        df = pd.DataFrame({
            "valid_time": load.index,
            "system_load": load.values.squeeze(),
            "solar_generation": solar.values.squeeze(),
            "wind_offshore_generation": wind_offshore.values.squeeze(),
            "wind_onshore_generation": wind_onshore.values.squeeze(),
        })

        df["valid_time"] = pd.to_datetime(df["valid_time"], utc=True)

        df["snapshot"] = now
        df["issue_time"] = now

        df["data_source"] = DataSource.ENTSOE.value
        df["data_type"] = DataType.OBSERVATION.value

        df["power_unit"] = PowerUnit.MW.value

        df = df[
            [
                "snapshot",
                "issue_time",
                "valid_time",
                "data_source",
                "data_type",
                "system_load",
                "solar_generation",
                "wind_offshore_generation",
                "wind_onshore_generation",
                "power_unit",
            ]
        ]

        df = df.sort_values("valid_time").reset_index(drop=True)

        return df