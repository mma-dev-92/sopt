import numpy as np
from src.degradation_model.constants import Constants
from src.preprocess.model import InputData, StorageStaticParams


class CalendarDegradationModel:
    """
    Simulates battery capacity loss due to calendar aging (time-at-rest).

    The model follows a semi-empirical approach where degradation_model is the product
    of a time-diffusion process (sqrt of time), temperature stress (Arrhenius),
    and State of Charge (SOC) stress.
    """

    def __init__(self, storage_static_params: StorageStaticParams, dt: float) -> None:
        """
        Initializes the model and pre-calculates the base diffusion coefficient.
        """
        self.sp = storage_static_params
        # Renamed to reflect physical 'k' factor in the diffusion model
        self.diffusion_coeff = self._calculate_diffusion_coefficient(self.sp, dt)

    def degradation(self, temperature: np.ndarray, soc: np.ndarray, storage_age: np.ndarray) -> float:
        """
        Calculates the cumulative capacity loss increment for the given period.

        Args:
            temperature: Ambient temperature time-series in Kelvin [K].
            soc: State of Charge time-series as coefficients [0, 1].
            storage_age: Storage age described in number of timesteps.

        Returns:
            float: Total fractional capacity loss increment for the period (e.g., 1.5e-5).
        """
        temperature_deg = self._temperature_degradation(temperature)
        soc_deg = self._soc_degradation(soc)
        time_deg = self._time_decay_degradation(storage_age)

        # The product of stressors represents the instantaneous degradation_model rate
        return (temperature_deg * soc_deg * time_deg).sum()

    @staticmethod
    def _calculate_diffusion_coefficient(sp: StorageStaticParams, dt: float) -> float:
        """
        Calculates the base 'k' coefficient for the sqrt(t) degradation_model curve.

        This coefficient scales the model such that the battery reaches its
        max_capacity_loss exactly at time_decay_duration_years under
        reference conditions.

        Returns:
            float: The normalized diffusion rate per timestep.
        """
        cal_deg_time = sp.degradation.time_decay_duration_years
        max_cap_loss = sp.degradation.max_capacity_loss

        # Total number of steps in the theoretical calendar life
        total_steps = cal_deg_time * 365 * 24 / dt
        return max_cap_loss / np.sqrt(total_steps)

    def _temperature_degradation(self, temperature: np.ndarray) -> np.ndarray:
        """
        Calculates the temperature stress factor using the Arrhenius equation.

        Maps the current cell temperature (ambient + offset) to a multiplier
        relative to the reference temperature.
        """
        activation_energy = self.sp.deg_model.activation_energy
        reference_temperature = self.sp.deg_model.reference_temperature
        temperature_offset = self.sp.deg_model.temperature_offset

        return np.exp(
            (activation_energy / Constants.R)
            *
            (1 / reference_temperature - 1 / (temperature + temperature_offset))
        )

    def _soc_degradation(self, soc: np.ndarray) -> np.ndarray:
        """
        Calculates the SOC stress factor using an exponential relationship.

        High SOC levels increase the chemical potential, accelerating
        side-reactions like SEI layer growth.
        """
        soc_sensitivity = self.sp.deg_model.soc_sensitivity

        return np.exp(soc_sensitivity * soc)

    def _time_decay_degradation(self, nth_timestep: np.ndarray) -> np.ndarray:
        """
        Calculates the incremental time-decay component (d/dt of sqrt(t)).

        Reflects that degradation_model slows down over time as the SEI layer thickens,
        making further diffusion more difficult.
        """
        # Safety: Ensure nth_timestep is never 0 to avoid division by zero
        safe_time = np.maximum(nth_timestep, 1e-6)
        return self.diffusion_coeff / (2 * np.sqrt(safe_time))