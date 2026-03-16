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
        rev = self.variables.rev
        return cp.sum(rev)