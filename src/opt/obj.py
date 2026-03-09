from abc import abstractmethod, ABCMeta

import cvxpy as cp

from src.opt.indices import Indices
from src.opt.parameters import Parameters
from src.opt.variables import Variables


class ObjectiveGenerator(metaclass=ABCMeta):

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

        gen_eta = self.params.storage_params.gen_efficiency
        price = self.params.prices.squeeze()

        return cp.sum(cp.multiply(price, (gen_eta * gen - load)))