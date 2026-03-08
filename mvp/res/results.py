import dataclasses
from datetime import date
from typing import Self

import pandas as pd
import numpy as np

from mvp.opt.engine import Engine


@dataclasses.dataclass
class OptResults:
    obj: float
    """objective function value"""
    partition: date
    """opt problem partition"""
    cum_rev: pd.Series
    """cumulative revenue (for debugging)"""
    load: pd.Series
    """load variable dump"""
    gen: pd.Series
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

        idx = engine.indices.t_idx.vals
        variables = engine.variables
        params = engine.parameters

        soc = np.asarray(variables.soc.value).flatten()
        gen = np.asarray(variables.gen.value).flatten()
        load = np.asarray(variables.load.value).flatten()

        soc_s = pd.Series(soc, index=idx, name="soc")
        gen_s = pd.Series(gen, index=idx, name="gen")
        load_s = pd.Series(load, index=idx, name="load")

        price = params.prices.squeeze()
        dt = params.dt
        gen_eta = params.storage_params.gen_efficiency

        rev = dt * price * (gen_eta * gen - load)
        cum_rev = pd.Series(np.cumsum(rev), index=idx, name="cum_rev")

        return cls(
            obj=rev.sum(),
            partition=engine.input_data.params.partition,
            cum_rev=cum_rev,
            load=load_s,
            gen=gen_s,
            soc=soc_s,
        )
