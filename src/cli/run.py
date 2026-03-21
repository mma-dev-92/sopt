from pathlib import Path
import click
import pandas as pd

from src.opt.engine import Engine
from src.postprocess.export import export_opt_results
from src.postprocess.results import OptResults
from src.preprocess.config import Configuration
from src.preprocess.load import load_input_data
from src.preprocess.model import InputData


def run_opt(input_data: InputData) -> pd.DataFrame:
    engine = Engine(input_data)
    engine.build()
    engine.optimize()
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
    export_opt_results(input_data, configuration, results_df)
