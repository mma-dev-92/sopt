from typing import Protocol, Any

import numpy as np

from src.opt.engine import Engine
from src.rl.spaces import Action, Observation, TransitionResult
from src.rl.reward import RewardCalculator

# TODO: change implementation of opt problem (use cp.Parameter) and let it run each step and update itself

class EnvironmentInterface(Protocol):
    """
    Strict contract for the simulation environment.
    Any external script interacting with the battery MUST use these methods.
    """
    def reset(self) -> Observation:
        ...

    # TODO: market data is just a price vector
    def step(self, action: Action, price_vector: np.ndarray) -> TransitionResult:
        ...


class BatteryEnvironment(EnvironmentInterface):
    """
    Concrete implementation of the simulation environment.
    Orchestrates the Optimizer and the physical BatteryInterface.
    """
    def __init__(self, engine: Engine, max_days: int = 3650) -> None:
        pass

    def step(self, action: Action, price_vector: np.ndarray) -> TransitionResult:
        pass

