from pydantic import BaseModel
from datetime import datetime


class Params(BaseModel):
    timestep_start: datetime
    timestep_end: datetime
