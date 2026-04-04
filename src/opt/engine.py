from typing import Type

import cvxpy as cp
import numpy as np

import src.opt.constraints as constr
import src.opt.obj as obj
from src.opt.indices import TimeIndex
from src.opt.parameters import Parameters
from src.opt.variables import Variables
from src.preprocess.model import InputData, StorageStaticParams


class Engine:

    _constraint_generators: list[Type[constr.ConstraintGenerator]] = [
        constr.BalancingConstraintGenerator,
        constr.PowerConstraintGenerator,
        constr.CapacityConstraintGenerator,
        constr.DecisionConstraintGenerator,
        constr.RevenueDefinitionConstraintGenerator,
    ]

    _obj_generators: list[Type[obj.ObjectiveGenerator]] = [
        obj.RevenueObjectiveGenerator,
        obj.LambdaPenaltyObjectiveGenerator,
    ]

    def __init__(self, storage_static_params: StorageStaticParams, dt: float, n_hours: int = 24) -> None:
        self.opt_problem: cp.Problem | None = None

        self.t_idx = TimeIndex(dt, n_hours)
        self.parameters = Parameters.create(storage_static_params, self.t_idx)
        self.variables = Variables.create(self.t_idx)

    def build(self) -> None:
        constraints = self._build_constraints()
        objective = self._build_obj()
        self.opt_problem = cp.Problem(objective, constraints)

    def _build_constraints(self) -> list[cp.Constraint]:
        result = []
        for constraint_generator_type in self._constraint_generators:
            generator = constraint_generator_type(
                self.t_idx, self.variables, self.parameters
            )
            result.extend(generator.generate())

        return result

    def _build_obj(self) -> cp.Minimize | cp.Maximize:
        result = 0
        for obj_generator_type in self._obj_generators:
            generator = obj_generator_type(
                self.parameters,
                self.variables,
            )
            result += generator.generate()

        return cp.Maximize(result)

    def optimize(self, debug: bool = False) -> None:
        self.opt_problem.solve(solver=cp.HIGHS, verbose=debug)
        if self.opt_problem.status not in ("optimal", "optimal_inaccurate"):
            raise RuntimeError(self.opt_problem.status)

    def update_dynamic_parameters(
            self,
            lambda_penalty: np.ndarray,
            energy_price: np.ndarray,
            init_soc: float,
            cap: float
    ) -> None:
        assert lambda_penalty.shape == energy_price.shape == self.t_idx.vals.shape
        dp = self.parameters.dynamic

        dp.lambda_penalty.value = lambda_penalty
        dp.energy_price.value = energy_price
        dp.init_soc.value = init_soc
        dp.cap.value = cap
