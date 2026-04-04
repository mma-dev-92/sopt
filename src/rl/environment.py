from typing import Protocol

import numpy as np
import pandas as pd

from src.degradation_model.calendar import CalendarDegradationModel
from src.degradation_model.constants import Constants
from src.degradation_model.cycle import CycleDegradationModel
from src.opt.engine import Engine
from src.postprocess.opt_results import OptResults
from src.preprocess.model import StorageStaticParams
from src.rl.spaces import Action, Observation, State, TransitionResult


class EnvironmentInterface(Protocol):
    """
    Strict contract for the simulation environment.
    Any external script interacting with the battery MUST use these methods.
    """
    def reset(self) -> Observation:
        ...

    def step(self, action: Action) -> TransitionResult:
        ...


class BatteryEnvironment(EnvironmentInterface):

    def __init__(
            self,
            storage_static_params: StorageStaticParams,
            energy_prices: pd.Series,
            dt: float = 0.25,
            n_hours: int = 24,
            max_days: int = 3650,
    ) -> None:

        # initialize optimization engine
        self.storage_static_params = storage_static_params
        self.engine = Engine(storage_static_params, dt=dt, n_hours=n_hours)
        self.engine.build()

        # initialize capacity degradation models
        self.cycle_degradation_model = CycleDegradationModel(storage_static_params)
        self.calendar_degradation_model = CalendarDegradationModel(storage_static_params, dt)

        # compute initial state
        self.state_history: list[State] = []

        # initialize observation dataset
        # (this dataset will contain data for the whole episode, in each iteration it will be sampled)
        self.prices = energy_prices.copy()

        # set maximal episode length (in days)
        self.max_days = max_days

    def step(self, action: Action) -> TransitionResult:
        # get current state and observation
        current_state = self.state_history[-1]
        current_observation = self.get_observation(current_state)

        # update optimization engine
        self.engine.update_dynamic_parameters(
            lambda_penalty=action.lambda_param,
            energy_price=current_observation.energy_prices,
            init_soc=current_state.soc,
            cap=current_state.cap,
        )

        # run the optimization
        self.engine.optimize()

        # fetch optimization results
        results = OptResults.from_engine(self.engine)

        # compute next state and save it
        self.state_history.append(State(
            cap=current_state.cap - self.compute_degradation(results),
            soc=results.soc.values[-1],
            nth_day=current_state.nth_day + 1
        ))

        return TransitionResult(
            observation=self.get_observation(self.state_history[-1]),
            reward=results.obj,
            terminated=self.is_terminated(),
            # TODO: @Kuba: just add here whatever you will need for debugging
            info=dict(),
        )

    def reset(self) -> Observation:
        # reset state history
        self.state_history = []
        self.state_history.append(self.initial_state(self.storage_static_params))

        return self.get_observation(self.state_history[0])

    def is_terminated(self) -> bool:

        last_state = self.state_history[-1]
        max_cap_loss = self.storage_static_params.degradation.max_capacity_loss
        init_cap = self.storage_static_params.technical.capacity

        cap_degradation_constr = last_state.cap / init_cap < max_cap_loss
        episode_len_constr = last_state.nth_day < self.max_days

        return not (cap_degradation_constr and episode_len_constr)

    @staticmethod
    def initial_state(storage_static_params: StorageStaticParams) -> State:
        init_cap = storage_static_params.technical.capacity
        min_rel_soc = storage_static_params.technical.soc_limits.min

        return State(cap=init_cap, soc=min_rel_soc * init_cap, nth_day=0)

    def get_observation(self, state: State) -> Observation:
        timesteps_per_day = len(self.engine.t_idx)
        nth_day = state.nth_day

        price_vector = self.prices.iloc[nth_day*timesteps_per_day:(nth_day+1)*timesteps_per_day].values
        return Observation(energy_prices=price_vector)

    def compute_degradation(self, results: OptResults) -> float:
        temperature = np.ones(len(self.engine.t_idx)) * Constants.opt_temperature
        nth_state = self.state_history[-1]

        current_capacity = nth_state.cap
        nth_day = nth_state.nth_day

        soc_t = results.soc / current_capacity
        dt = self.engine.t_idx.dt

        cycle_degradation = self.cycle_degradation_model.degradation(
            cycle_t=soc_t,
            temperature_t=temperature
        )
        calendar_degradation = self.calendar_degradation_model.degradation(
            temperature=temperature,
            soc=soc_t,
            storage_age=nth_day * 24 * dt
        )

        return (cycle_degradation + calendar_degradation) * nth_state.cap
