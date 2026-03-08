from pathlib import Path
import click

from mvp.opt.engine import Engine
from mvp.preprocess.config import Configuration
from mvp.preprocess.load import load_input_data
from mvp.res.results import OptResults


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
    results = OptResults.from_engine(engine)
