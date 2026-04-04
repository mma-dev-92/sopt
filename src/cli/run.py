from pathlib import Path
import click
import numpy as np
import pandas as pd

from src.opt.engine import Engine
from src.postprocess.opt_results import OptResults
from src.preprocess.config import Configuration
from src.preprocess.load import load_input_data
from src.preprocess.model import InputData

def extract_energy_prices(
    input_data: InputData,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None
) -> np.ndarray:
    start = start or input_data.params.timestep_start
    end = end or input_data.params.timestep_end
    dt = input_data.market_data.resolution
    date_range = pd.date_range(start=pd.to_datetime(start), end=pd.to_datetime(end), freq=f"{int(60 * dt)}min")[:-1]
    return input_data.market_data.prices[date_range].values


def run_opt(input_data: InputData) -> pd.DataFrame:
    engine = Engine(input_data)
    engine.build()
    engine.optimize(
        lambda_penalty=np.ones(engine.t_idx.t_idx.vals.shape),
        energy_price=extract_energy_prices(input_data),
        init_soc=input_data.storage_state_params.soc,
        cap=input_data.storage_static_params.technical.capacity,
    )
    results = OptResults.from_engine(engine)
    return results.to_dataframe()


@click.command()
@click.option(
    '--config',
    '-c',
    required=True,
    type=click.Path(exists=True),
)
def cli_run(config: Path) -> None:
    configuration = Configuration.load(Path(config))
    input_data = load_input_data(configuration)
    results_df = run_opt(input_data)
