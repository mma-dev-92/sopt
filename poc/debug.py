from src.preprocess.load import load_storage_static_params

import os
from pathlib import Path

import pandas as pd
import numpy as np

from src.preprocess.model import StorageStaticParams, InputData

assets_dir = Path('/home/maciek/Projects/sopt/assets')
storage_static_params_path = assets_dir / 'input' / 'storage_static_params.yaml'

storage_static_params = load_storage_static_params(storage_static_params_path)


# %% md
#### Calendar degradation_model model
# %%
class CalendarDegradationModel:
    _R = 8.314  # universal gas constant [J/mol]
    _T_ref = 298.15  # reference temperature [K]
    _dT_init = 3.5  # constant offset (bess is warmer than the outside temp) [K] - it should depend on the storage power(t); future improvement
    _E_a = 5 * 1e4  # activation energy [J/mol]

    _SOC_DEG_RATE = 0.6

    def __init__(self, input_data: InputData) -> None:
        dt = input_data.market_data.resolution
        self.base_ref = self._compute_base_ref(input_data.storage_static_params, dt)

    def degradation(self, temperature: float, soc: float, nth_day: float) -> float:
        temperature_deg = self._temperature_degradation(temperature)
        soc_deg = self._soc_degradation(soc)
        time_deg = self._time_decay_degradation(nth_day)

        return temperature_deg * soc_deg * time_deg

    @staticmethod
    def _compute_base_ref(sp: StorageStaticParams, dt: float) -> float:
        """
        Base reference degradation_model rate for time decay (per timestep)
        """
        cal_deg_time = sp.degradation.time_decay_duration_years  # calendar decay time
        max_cap_loss = sp.degradation.max_capacity_loss  # end of life capacity

        return max_cap_loss / (np.sqrt(cal_deg_time * 365 * 24 / dt))

    def _temperature_degradation(self, temperature: float) -> float:
        """
        Temperature degradation_model component
        """
        return np.exp(
            (self._E_a / self._R) * (1 / self._T_ref - 1 / (temperature + self._dT_init))
        )

    def _soc_degradation(self, soc: float) -> float:
        """
        State of charge degradation_model component
        """
        return np.exp(self._SOC_DEG_RATE * soc)

    def _time_decay_degradation(self, nth_timestep: float) -> float:
        """
        Time decay component
        """
        return self.base_ref / (2 * np.sqrt(nth_timestep))


# %% md
#### Cycle degradation_model model
# %%
class CycleDegradationModel:

    def __init__(self, input_data: InputData) -> None:
        self.alpha = self._compute_alpha(input_data)
        self.soc_rel_lb = indata.storage_static_params.technical.soc_limits.min

    def degradation(self, soc_max: float, cap_t: float) -> float:
        return self.alpha * soc_max * self._F((soc_max - self.soc_rel_lb * cap_t) / cap_t)

    @staticmethod
    def _F(relative_depth: float) -> float:
        """
        cycle-depth degradation_model function
        """
        return relative_depth ** 1.3

    @staticmethod
    def _compute_alpha(input_data: InputData) -> float:
        """
        compute alpha parameter: estimated degradation_model per 1 unit of cycle soc_max
        """
        sp = input_data.storage_static_params
        max_cap_loss = sp.degradation.max_capacity_loss  # e.g., 0.2
        n_cycles = sp.degradation.n_cycles  # e.g., 5000
        dod_avg = sp.degradation.dod_avg  # e.g., 0.8

        # The "Reference Stress" of a standard cycle
        # This must match the exponent in your _F function (1.3)
        reference_stress = CycleDegradationModel._F(dod_avg)

        # ALPHA = (Total Fraction to Lose) / (Total Energy Throughput * Stress at that Throughput)
        # This ensures that one cycle at dod_avg results exactly in (max_cap_loss / n_cycles)
        return max_cap_loss / (n_cycles * dod_avg * reference_stress)

    @staticmethod
    def _compute_total_energy_throughput(input_data: InputData) -> float:
        """
        Average (simplified) formula to estimate total energy throughput
        """
        dod_avg = input_data.storage_static_params.degradation.dod_avg
        n_cycles = input_data.storage_static_params.degradation.n_cycles
        cap_0 = input_data.storage_static_params.technical.capacity
        max_cap_loss = input_data.storage_static_params.degradation.max_capacity_loss

        cap_eol = cap_0 * (1 - max_cap_loss)

        return dod_avg * n_cycles * (cap_0 + cap_eol) / 2

# %%


class RandomCycleGenerator:
    def __init__(
            self,
            input_data: InputData,
            dod_mean: float = 0.7,
            dod_std: float = 0.07,
            max_time_gamma_shape: float = 3.0,
            max_time_gamma_scale: float = 0.5,
            cycle_start_min: int = 8,
            cycle_start_max: int = 20,
    ) -> None:
        # time resolution
        self.dt = input_data.market_data.resolution
        # storage params
        self.charge_eta = input_data.storage_static_params.technical.charge_efficiency
        self.discharge_eta = input_data.storage_static_params.technical.discharge_efficiency
        self.dod_min = input_data.storage_static_params.technical.soc_limits.min
        self.cap_init = input_data.storage_static_params.technical.capacity
        self.nom_p = input_data.storage_static_params.technical.power
        # dod generation parameters
        self.dod_mean = dod_mean
        self.dod_std = dod_std
        # max_time generation parameters
        self.max_time_gamma_shape = max_time_gamma_shape
        self.max_time_gamma_scale = max_time_gamma_scale
        # possible starting timestep of a cycle
        self.cycle_start_min = cycle_start_min
        self.cycle_start_max = cycle_start_max

    def draw_random_cycle(self, cap_t: float, nom_p: float) -> np.ndarray:
        # 1. Draw cycle intensity
        dod_max = np.clip(np.random.normal(loc=self.dod_mean, scale=self.dod_std), self.dod_min + 0.05, 1.0)
        delta_soc = dod_max - self.dod_min

        # 2. Calculate durations in timesteps (Charge/Discharge at P_max)
        t_charge = int(np.ceil((delta_soc * cap_t) / (nom_p * self.dt * self.charge_eta)))
        t_plateau = int(
            np.round(np.random.gamma(shape=self.max_time_gamma_shape, scale=self.max_time_gamma_scale) / self.dt))
        t_discharge = int(np.ceil((delta_soc * cap_t) / (nom_p * self.dt)))

        steps_per_day = int(24 / self.dt)

        # 3. SAFETY MECHANISM: Ensure the cycle fits in the 24h window
        # Calculate minimum required steps for ramps only
        min_required_steps = t_charge + t_discharge

        # Draw start_idx, but cap it so at least the ramps can fit
        latest_possible_start = steps_per_day - min_required_steps
        start_idx = int(
            np.random.uniform(
                low=self.cycle_start_min,
                high=min(self.cycle_start_max, latest_possible_start)
            )
        )

        # Calculate available space for the plateau
        available_for_plateau = steps_per_day - start_idx - min_required_steps

        # Shrink plateau if it doesn't fit in the remaining gap
        if t_plateau > available_for_plateau:
            t_plateau = max(0, available_for_plateau)

        # 4. Create and Fill the full series
        soc_series = np.full(steps_per_day, self.dod_min)

        # Calculate phase end indices
        c_end = start_idx + t_charge
        p_end = c_end + t_plateau
        d_end = p_end + t_discharge

        # Charge Ramp
        idx_c = np.arange(start_idx, c_end)
        if len(idx_c) > 0:
            soc_series[idx_c] = np.linspace(self.dod_min, dod_max, len(idx_c))

        # Plateau
        soc_series[c_end: p_end] = dod_max

        # Discharge Ramp
        idx_d = np.arange(p_end, d_end)
        if len(idx_d) > 0:
            soc_series[idx_d] = np.linspace(dod_max, self.dod_min, len(idx_d))

        return soc_series


# %% md
#### Load Temperature Data
# %%
from poc.data.bronze.dataset import WeatherDataset
from poc.data.bronze.schema import WeatherSchema
from poc.data.bronze.metadata import BronzeBaseDatasetMetadata

# 10-years time horizon
start = pd.Timestamp("2016-01-01", tz="UTC")
end = pd.Timestamp("2026-01-01", tz="UTC")

weather_metadata = BronzeBaseDatasetMetadata.from_yaml(
    "/home/maciek/Projects/sopt/poc/assets/datasets/weather.yaml",
    schema=WeatherSchema,
)

# calendar killer - Spain, Sevilla
sevilla_weather_dataset = WeatherDataset(
    metadata=weather_metadata,
    start=start,
    end=end,
    latitude=37.3891,
    longitude=-5.9845,
)

# control group - Kiruna, Sweden
kiruna_weather_dataset = WeatherDataset(
    metadata=weather_metadata,
    start=start,
    end=end,
    latitude=67.8558,
    longitude=-5.9845,
)

# extreme stress - Dubai, UAE
dubai_weather_dataset = WeatherDataset(
    metadata=weather_metadata,
    start=start,
    end=end,
    latitude=25.2048,
    longitude=55.2708,
)

# high solar profile - Casablanca, Morocco
casablanca_weather_dataset = WeatherDataset(
    metadata=weather_metadata,
    start=start,
    end=end,
    latitude=33.5731,
    longitude=-7.5898
)

# %%
C_TO_K = 273.15

ambient_temperature = dict(
    sevilla=sevilla_weather_dataset.load().set_index('valid_time')['temperature'] + C_TO_K,
    kiruna=kiruna_weather_dataset.load().set_index('valid_time')['temperature'] + C_TO_K,
    dubai=dubai_weather_dataset.load().set_index('valid_time')['temperature'] + C_TO_K,
    casablanca=casablanca_weather_dataset.load().set_index('valid_time')['temperature'] + C_TO_K,
)
# %%
from src.preprocess.load import load_input_data, load_market_data, load_storage_state_params, load_params

input_dir = assets_dir / 'input'
indata = InputData(
    market_data=load_market_data(input_dir / '2025_DA_prices.csv'),
    storage_static_params=load_storage_static_params(input_dir / 'storage_static_params.yaml'),
    storage_state_params=load_storage_state_params(input_dir / 'storage_state.yaml'),
    params=load_params(input_dir / 'params.yaml'),
)

indata.market_data.resolution = 1.0
# %%
from abc import ABC, abstractmethod


class BaseDegradationRunner(ABC):
    def __init__(
            self,
            input_data: InputData,
            weather_data: pd.Series,
            seed: int = 42
    ):
        self.input_data = input_data
        self.weather_data = weather_data
        self.seed = seed

        # Physics & Time Constants
        self.dt = input_data.market_data.resolution
        self.steps_per_day = int(24 / self.dt)
        self.cap_0 = input_data.storage_static_params.technical.capacity
        self.nom_p = input_data.storage_static_params.technical.power
        self.cap_eol = self.cap_0 * (1 - input_data.storage_static_params.degradation.max_capacity_loss)

        # Stochastic Control: Use a local Generator instance
        self.rng = np.random.default_rng(seed)
        self.cycle_gen = RandomCycleGenerator(input_data)
        # Patch the generator to use our seeded RNG for reproducibility
        self.cycle_gen.rng = self.rng

    @abstractmethod
    def compute_daily_degradation(self, soc_day: np.ndarray, temp_day: np.ndarray, day_idx: int, cap_t: float) -> dict:
        """
        Calculates total losses for one day.
        Must return: {'cycle_loss': float, 'calendar_loss': float}
        """
        pass

    def run(self) -> pd.DataFrame:
        cap_t = self.cap_0
        daily_results = []
        total_days = len(self.weather_data) // self.steps_per_day

        for day_idx in range(total_days):
            if cap_t <= self.cap_eol:
                break

            # 1. Generate the SOC profile for the current state
            soc_day = self.cycle_gen.draw_random_cycle(cap_t, self.nom_p)

            # 2. Get the weather slice
            start_idx = day_idx * self.steps_per_day
            temp_day = self.weather_data.iloc[start_idx: start_idx + self.steps_per_day].values

            # 3. Compute degradation_model via the abstract implementation
            losses = self.compute_daily_degradation(soc_day, temp_day, day_idx, cap_t)

            cyc_l = losses.get('cycle_loss', 0.0)
            cal_l = losses.get('calendar_loss', 0.0)

            # 4. Update State
            cap_t -= (cyc_l + cal_l)

            # 5. Log Daily Metrics
            daily_results.append({
                'day': day_idx + 1,
                'date': self.weather_data.index[start_idx],
                'capacity': cap_t,
                'soh': cap_t / self.cap_0,
                'cycle_loss': cyc_l,
                'calendar_loss': cal_l,
                'avg_temp': np.mean(temp_day),
                'max_soc': np.max(soc_day)
            })

        return pd.DataFrame(daily_results).set_index('day')


class GroundTruthRunner(BaseDegradationRunner):
    def __init__(self, input_data, weather_data, seed=42):
        super().__init__(input_data, weather_data, seed)
        self.cal_model = CalendarDegradationModel(input_data)
        self.cyc_model = CycleDegradationModel(input_data)
        self.soc_min = input_data.storage_static_params.technical.soc_limits.min

    def compute_daily_degradation(self, soc_day, temp_day, day_idx, cap_t):
        # 1. Calendar: Sum the individual timestep losses
        calendar_loss = 0.0
        start_step_global = day_idx * self.steps_per_day

        for i in range(self.steps_per_day):
            step_loss = self.cal_model.degradation(
                temperature=temp_day[i],
                soc=soc_day[i],
                nth_day=start_step_global + i + 1
            )
            calendar_loss += step_loss

        # 2. Cycle: Single calculation for the day
        soc_peak = np.max(soc_day)
        delta_soc = max(0, soc_peak - self.soc_min)
        cycle_loss = self.cyc_model.degradation(delta_soc, cap_t)

        return {
            'cycle_loss': cycle_loss,
            'calendar_loss': calendar_loss
        }


# %%
runners = dict()
for city in ambient_temperature:
    runners[city] = GroundTruthRunner(
        input_data=indata,
        weather_data=ambient_temperature[city],
        seed=42,
    )
# %%
results = dict()
for city, runner in runners.items():
    results[city] = runner.run()
# %%
results.keys()
# %%
calendar_losses = pd.concat([results[city]['calendar_loss'] for city in results], axis=1)
calendar_losses.columns = list(results.keys())
(1 - calendar_losses.cumsum(axis=0)).plot(title='Cumulative Calendar Losses')
# %%
cycle_losses = pd.concat([results[city]['cycle_loss'] for city in results], axis=1)
cycle_losses.columns = list(results.keys())
(1 - cycle_losses.cumsum(axis=0)).plot(title='Cumulative Cycle Losses')
