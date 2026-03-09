import pandas as pd
import numpy as np

from src.opt.engine import Engine
from src.postprocess.results import OptResults
from src.preprocess.config import Configuration


def clean_solver_noise(
    df: pd.DataFrame,
    tol: float = 1e-6,
    decimals: int = 6
) -> pd.DataFrame:
    """
    Clean numerical noise from optimization results.

    - Values with |x| < tol → 0
    - Values close to integers → integer
    - Final rounding to given decimals
    """

    out = df.copy()
    # remove very small values
    out = out.mask(out.abs() < tol, 0)
    # snap values close to integers
    out = np.where(np.isclose(out, np.round(out), atol=tol), np.round(out), out)
    # convert back to DataFrame
    out = pd.DataFrame(out, index=df.index, columns=df.columns)
    # final rounding
    return out.round(decimals)


def create_exportable_results(engine: Engine) -> pd.DataFrame:
    results = OptResults.from_engine(engine)
    price_vector = engine.input_data.market_data.prices.loc[engine.indices.t_idx.vals, :]

    result = pd.concat([price_vector, results.cum_rev, results.gen, results.load, results.soc], axis=1)
    return clean_solver_noise(result)


def export_opt_results(engine: Engine, configuration: Configuration) -> None:

    df = create_exportable_results(engine)
    partition = engine.input_data.params.partition
    ext = configuration.output_paths.format
    export_path = configuration.output_paths.result_dir / f"{partition}.{ext}"

    match ext:
        case "parquet":
            df.to_parquet(export_path)
        case "csv":
            df.to_csv(export_path)
        case "xlsx":
            df.to_excel(export_path)
        case _:
            raise ValueError(f"Unrecognized export format: {ext}")
