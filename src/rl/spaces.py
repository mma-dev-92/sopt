from dataclasses import dataclass

@dataclass
class Action:
    """
    The control signal provided by the external controller/AI.
    """
    lambda_param: float


@dataclass
class Observation:
    """
    The state of the world as seen by the external controller.
    """
    nth_day: int
    capacity_loss: float
    # In the future, you can easily add: market_price_avg, temperature_forecast, etc.


@dataclass
class TransitionResult:
    """
    The full package returned after one simulation step.
    """
    observation: Observation
    reward: float
    is_done: bool
