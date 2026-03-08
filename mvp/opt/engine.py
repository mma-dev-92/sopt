from typing import Type

import cvxpy as cp

import mvp.opt.constraints as constr
import mvp.opt.obj as obj
from mvp.opt.indices import Indices
from mvp.opt.parameters import Parameters
from mvp.opt.variables import Variables
from mvp.preprocess.model import InputData


class Engine:

    _constraint_generators: list[Type[constr.ConstraintGenerator]] = [
        constr.BalancingConstraintGenerator,
        constr.PowerConstraintGenerator,
        constr.CapacityConstraintGenerator,
    ]

    _obj_generators: list[Type[obj.ObjectiveGenerator]] = [
        obj.RevenueObjectiveGenerator,
    ]

    def __init__(self, input_data: InputData) -> None:
        self.input_data = input_data

        self.opt_problem: cp.Problem | None = None

        self.indices = Indices.create(input_data)
        self.parameters = Parameters.create(input_data, self.indices)
        self.variables = Variables.create(self.indices)


    def build(self) -> None:
        constraints = self._build_constraints()
        objective = self._build_obj()
        self.opt_problem = cp.Problem(objective, constraints)

    def _build_constraints(self) -> list[cp.Constraint]:
        result = []
        for constraint_generator_type in self._constraint_generators:
            generator = constraint_generator_type(
                self.indices, self.variables, self.parameters
            )
            result.extend(generator.generate())

        return result

    def _build_obj(self) -> cp.Minimize | cp.Maximize:
        result = 0.0
        for obj_generator_type in self._obj_generators:
            generator = obj_generator_type(
                self.indices,
                self.parameters,
                self.variables,
            )
            result += generator.generate()

        return cp.Maximize(result)

    def optimize(self) -> None:
        self.opt_problem.solve(
            solver=cp.HIGHS,
            verbose=False,
        )
        if self.opt_problem.status not in ("optimal", "optimal_inaccurate"):
            raise RuntimeError(self.opt_problem.status)