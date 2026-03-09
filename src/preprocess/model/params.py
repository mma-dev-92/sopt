from pydantic import BaseModel
from datetime import date


class Params(BaseModel):
    partition: date
