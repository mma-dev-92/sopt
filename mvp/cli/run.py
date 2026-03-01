from pathlib import Path
import click

from mvp.opt.engine import Engine
from mvp.preprocess.config import Configuration
from mvp.preprocess.load import load_input_data
from mvp.preprocess.model import InputData


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
    engine = Engine(input_data, configuration)
    engine.build()
    engine.optimize()
