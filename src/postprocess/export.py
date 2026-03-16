import pandas as pd
import numpy as np
import cvxpy as cp

from src.opt.engine import Engine
from src.opt.indices import Index
from src.preprocess.config import Configuration
from src.preprocess.model import InputData


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


def fetch_time_variable_from_opt_results(variable: cp.Variable, index: Index) -> pd.Series:
    return pd.Series(
        data=np.asarray(variable.value).flatten(),
        index=index.vals,
        name=variable.name,
    )


def create_exportable_results(engine: Engine) -> pd.DataFrame:
    price_vector = engine.input_data.market_data.prices.loc[engine.indices.t_idx.vals, :]

    decision_variable_results = [
        fetch_time_variable_from_opt_results(variable, engine.indices.t_idx)
        for variable in engine.variables.decision_variables
    ]

    result = pd.concat([price_vector] + decision_variable_results, axis=1)
    return clean_solver_noise(result)


def export_opt_results(
        input_data: InputData,
        configuration: Configuration,
        results_df: pd.DataFrame
) -> None:

    partition = input_data.params.partition
    ext = configuration.output_paths.format
    export_path = configuration.output_paths.result_dir / f"{partition}.{ext}"

    match ext:
        case "parquet":
            results_df.to_parquet(export_path)
        case "csv":
            results_df.to_csv(export_path)
        case "xlsx":
            results_df.to_excel(export_path)
        case _:
            raise ValueError(f"Unrecognized export format: {ext}")
