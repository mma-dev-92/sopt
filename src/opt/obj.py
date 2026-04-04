from abc import abstractmethod, ABCMeta

import cvxpy as cp

from src.opt.parameters import Parameters
from src.opt.variables import Variables


class ObjectiveGenerator(metaclass=ABCMeta):

    def __init__(
        self,
        params: Parameters,
        variables: Variables,
    ) -> None:
        self.variables = variables
        self.params = params

    @abstractmethod
    def generate(self) -> cp.Expression:
        pass


class RevenueObjectiveGenerator(ObjectiveGenerator):
    def generate(self) -> cp.Expression:
        return cp.sum(self.variables.rev)


class LambdaPenaltyObjectiveGenerator(ObjectiveGenerator):
    def generate(self) -> cp.Expression:
        return -cp.sum(
            cp.multiply(
                self.params.dynamic.lambda_penalty,
                (self.variables.charge + self.variables.discharge) / 2.0
            )
        )