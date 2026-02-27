import cvxpy as cp
import numpy as np

from mvp.opt.indices import Indices
from mvp.opt.parameters import Parameters
from mvp.opt.variables import Variables
from mvp.preprocess.config import Configuration
from mvp.preprocess.model import InputData


class Engine:

    def __init__(self, input_data: InputData, config: Configuration) -> None:
        self.input_data = input_data
        self.config = config

        self.indices = Indices.create(input_data)
        self.parameters = Parameters.create(input_data)
        self.variables = Variables.create(input_data)

    def build(self) -> None:
        pass

    def optimize(self) -> None:
        pass
