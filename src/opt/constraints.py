import cvxpy as cp

from abc import abstractmethod, ABCMeta

from src.opt.indices import Indices
from src.opt.parameters import Parameters
from src.opt.variables import Variables


class ConstraintGenerator(metaclass=ABCMeta):

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
            self.variables.discharge,
            self.variables.charge,
        )

        charge_eta = self.params.storage_opt_params.charge_eta
        init_soc = self.params.storage_opt_params.init_soc

        return [
            soc[0] == init_soc,
            soc[1:] == soc[:-1] + charge_eta * load[:-1] - gen[:-1],
            soc[-1] == init_soc,
        ]


class PowerConstraintGenerator(ConstraintGenerator):
    def generate(self) -> list[cp.Constraint]:
        gen, load = (
            self.variables.discharge,
            self.variables.charge,
        )

        power = self.params.storage_opt_params.nom_p
        dt = self.params.dt

        return [gen + load <= dt * power]


class CapacityConstraintGenerator(ConstraintGenerator):
    def generate(self) -> list[cp.Constraint]:
        soc = self.variables.soc
        capacity = self.params.storage_opt_params.nom_cap
        soc_rel_lb = self.params.storage_opt_params.soc_rel_lb
        soc_rel_ub = self.params.storage_opt_params.soc_rel_ub

        return [
            soc >= soc_rel_lb * capacity,
            soc <= soc_rel_ub * capacity,
        ]


class DecisionConstraintGenerator(ConstraintGenerator):
    def generate(self) -> list[cp.Constraint]:

        gen = self.variables.discharge
        load = self.variables.charge

        bin_gen = self.variables.bin_gen
        bin_load = self.variables.bin_load

        return [
            bin_gen >= gen,
            bin_load >= load,
            bin_gen + bin_load <= 1,
        ]

class RevenueDefinitionConstraintGenerator(ConstraintGenerator):
    def generate(self) -> list[cp.Constraint]:

        gen = self.variables.discharge
        load = self.variables.charge
        rev = self.variables.rev

        gen_eta = self.params.storage_opt_params.discharge_eta
        price = self.params.prices.squeeze()

        return [rev == cp.multiply(price, (gen_eta * gen - load))]
