import dataclasses
from typing import Self

import cvxpy as cp
import pandas as pd
import numpy as np

from src.opt.engine import Engine


def clean_solver_noise(
        arr: np.ndarray,
        tol: float = 1e-6,
        decimals: int = 6
) -> np.ndarray:
    """
    Clean numerical noise from optimization results.

    - Values with |x| < tol → 0
    - Values close to integers → integer
    - Final rounding to given decimals
    """

    out = np.copy(arr)
    # remove very small values
    out[np.abs(out) < tol] = 0.0
    # snap values close to integers
    out = np.where(np.isclose(out, np.round(out), atol=tol), np.round(out), out)
    # round to decimals
    out = np.round(out, decimals)

    return out


@dataclasses.dataclass
class OptResults:
    obj: float
    """objective function value"""
    prices: pd.Series
    """energy price series"""
    rev: pd.Series
    """revenue per timestep (for debugging)"""
    charge: pd.Series
    """load variable dump"""
    discharge: pd.Series
    """gen variable dump"""
    soc: pd.Series
    """soc variable dump"""

    @classmethod
    def from_engine(cls, engine: Engine) -> Self:

        problem = engine.opt_problem
        if problem is None:
            raise RuntimeError("optimization problem not built")

        if problem.status not in ("optimal", "optimal_inaccurate"):
            raise RuntimeError(f"solver status: {problem.status}")

        idx = engine.t_idx.vals
        variables = engine.variables
        params = engine.parameters

        soc = cls.extract_time_variable_values(variables.soc, name="soc", idx=idx)
        charge = cls.extract_time_variable_values(variables.charge, name="charge", idx=idx)
        discharge = cls.extract_time_variable_values(variables.discharge, name="discharge", idx=idx)
        rev = cls.extract_time_variable_values(variables.rev, name="rev", idx=idx)
        prices = pd.Series(params.dynamic.energy_price.value, index=idx, name="price")

        return cls(
            obj=rev.sum(),
            rev=rev,
            prices=prices,
            charge=charge,
            discharge=discharge,
            soc=soc,
        )

    @staticmethod
    def extract_time_variable_values(variable: cp.Variable, name: str, idx: np.ndarray) -> pd.Series:
        extracted = clean_solver_noise(np.asarray(variable.value).flatten())
        return pd.Series(extracted, name=name, index=idx)

    def to_dataframe(self) -> pd.DataFrame:
        result = pd.concat([self.rev, self.prices, self.soc, self.charge, self.discharge], axis=1)
        result.index.name = "timestep"
        return result
