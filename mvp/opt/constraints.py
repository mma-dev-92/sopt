import cvxpy as cp

from abc import abstractmethod

from mvp.opt.indices import Indices
from mvp.opt.parameters import Parameters
from mvp.opt.variables import Variables


class ConstraintGenerator:

    def __init__(
            self,
            indices: Indices,
            variables: Variables,
            params: Parameters
    ) -> None:
        self.indices = indices
        self.params = params
        self.variables = variables

    @abstractmethod
    def generate(self) -> list[cp.Constraint]:
        pass


class BalancingConstraintGenerator(ConstraintGenerator):
    def generate(self) -> list[cp.Constraint]:
        soc, gen, load = (
            self.variables.soc,
            self.variables.gen,
            self.variables.load,
        )

        load_eta = self.params.storage_params.load_efficiency
        dt = self.params.dt
        init_soc = self.params.storage_params.init_soc

        return [
            soc[0] == init_soc,
            soc[1:] == soc[:-1] + (dt * load_eta) * load[:-1] - dt * gen[:-1],
            soc[-1] == init_soc,
        ]


class PowerConstraintGenerator(ConstraintGenerator):
    def generate(self) -> list[cp.Constraint]:
        gen, load = (
            self.variables.gen,
            self.variables.load,
        )

        power = self.params.storage_params.power
        dt = self.params.dt

        return [dt * (gen + load) <= power]


class CapacityConstraintGenerator(ConstraintGenerator):
    def generate(self) -> list[cp.Constraint]:
        soc = self.variables.soc
        capacity = self.params.storage_params.capacity
        dod = self.params.storage_params.dod

        return [
            soc >= dod.MIN * capacity,
            soc <= dod.MAX * capacity,
        ]