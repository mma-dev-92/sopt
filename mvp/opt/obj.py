from abc import abstractmethod

import cvxpy as cp

from mvp.opt.indices import Indices
from mvp.opt.parameters import Parameters
from mvp.opt.variables import Variables


class ObjectiveGenerator:

    def __init__(
        self,
        indices: Indices,
        params: Parameters,
        variables: Variables,
    ) -> None:
        self.indices = indices
        self.variables = variables
        self.params = params

    @abstractmethod
    def generate(self) -> cp.Expression:
        pass


class RevenueObjectiveGenerator(ObjectiveGenerator):
    def generate(self) -> cp.Expression:

        gen = self.variables.gen
        load = self.variables.load

        dt = self.params.dt
        gen_eta = self.params.storage_params.gen_efficiency
        price = self.params.prices.squeeze()

        return dt * cp.sum(cp.multiply(price, (gen_eta * gen - load)))