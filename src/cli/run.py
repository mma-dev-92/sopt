from pathlib import Path
import click
import numpy as np
import pandas as pd

from config import ROOT
from src.opt.engine import Engine
from src.postprocess.opt_results import OptResults
from src.preprocess.load import load_storage_static_params
from src.preprocess.model import StorageStaticParams


def run_opt(storage_static_params: StorageStaticParams, dt: float) -> pd.DataFrame:
    engine = Engine(storage_static_params, dt)
    engine.build()
    engine.update_dynamic_parameters(
        lambda_penalty=np.ones(len(engine.t_idx)),
        energy_price=np.random.random(size=len(engine.t_idx)),
        init_soc=storage_static_params.technical.capacity * storage_static_params.technical.soc_limits.min,
        cap=storage_static_params.technical.capacity,
    )
    engine.optimize(debug=True)
    results = OptResults.from_engine(engine)
    return results.to_dataframe()


def cli_run() -> None:
    storage_static_params = load_storage_static_params(Path(ROOT / 'assets' / 'input' / 'storage_static_params.yaml'))
    run_opt(storage_static_params, dt=0.5)
