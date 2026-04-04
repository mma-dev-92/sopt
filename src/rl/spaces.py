from dataclasses import dataclass

import numpy as np


@dataclass
class State:
    soc: float
    """current state of charge of the storage"""
    cap: float
    """current capacity of the storage"""
    nth_day: int
    """index of the optimized day"""


@dataclass
class Action:
    """
    The control signal provided by the external controller/AI.
    """
    lambda_param: np.ndarray


@dataclass
class Observation:
    """
    The state of the world as seen by the external controller.
    """
    energy_prices: np.ndarray
    ...  # later more features will appear here


@dataclass
class TransitionResult:
    """
    The full package returned after one (nth) simulation step.
    """
    observation: Observation
    """
    Observation for the next (n+1) step
    """
    reward: float
    """
    Reward for the last (n) step
    """
    terminated: bool
    """
    Is episode terminated
    """
    info: dict
    """
    Info (for debugging)
    """
