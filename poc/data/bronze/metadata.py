import enum
import yaml
import pandera.pandas as pa

from pydantic import BaseModel
from typing import Type


class TimeSemantics(str, enum.Enum):
    INTERVAL_START = "interval_start"
    INTERVAL_END = "interval_end"


class TimeResolution(str, enum.Enum):
    MIN15 = "15min"
    HOUR1 = "1h"


class BronzeBaseDatasetMetadata(BaseModel):

    dataset_name: str
    data_schema: Type[pa.DataFrameModel]
    timezone: str = "UTC"
    valid_time_resolution: TimeResolution
    time_semantics: TimeSemantics = TimeSemantics.INTERVAL_START
    description: str | None = None

    class Config:
        arbitrary_types_allowed = True

    @classmethod
    def from_yaml(cls, path: str, schema: Type[pa.DataFrameModel]):

        with open(path) as f:
            data = yaml.safe_load(f)

        return cls(**data, data_schema=schema)