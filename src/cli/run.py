from pathlib import Path
import click

from src.opt.engine import Engine
from src.postprocess.export import export_opt_results
from src.preprocess.config import Configuration
from src.preprocess.load import load_input_data


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
    engine = Engine(input_data)
    engine.build()
    engine.optimize()
    export_opt_results(engine, configuration)
