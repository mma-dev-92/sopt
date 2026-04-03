import numpy as np
import rainflow
from dataclasses import dataclass
from src.degradation_model.constants import Constants as C
from src.preprocess.model import InputData


@dataclass
class CycleParameters:
    """
    Represents extracted features of a single charge/discharge cycle.

    Attributes:
        dod (float): Depth of Discharge magnitude in energy units [MWh].
        soc_avg (float): Mean State of Charge during the cycle, normalized [0, 1] 
            relative to the technical operational limits.
        count (float): Rainflow cycle count (0.5 for half-cycles, 1.0 for full cycles).
        start_idx (int): The starting index of the cycle in the time-series array.
        end_idx (int): The ending index of the cycle in the time-series array.
    """
    dod: float
    soc_avg: float
    count: float
    start_idx: int
    end_idx: int


class CycleDegradationModel:
    """
    Computes battery capacity loss due to cyclic aging using a power-law 
    severity model coupled with an Arrhenius temperature stress factor.
    """

    def __init__(
            self,
            input_data: InputData,
            cycle_severity_exponent: float | None = None,
    ) -> None:
        """
        Initializes the model by solving for the curve-fit constant 'a'.

        Args:
            input_data: Configuration object containing technical and degradation_model params.
            cycle_severity_exponent: The 'b' coefficient in the power law. If None, 
                defaults to the constant specified in the global settings.
        """
        self._cycle_severity_exponent = cycle_severity_exponent or C.CYCLE_SEVERITY_EXPONENT
        self._a = self._compute_a(input_data, self._cycle_severity_exponent)
        self.sp = input_data.storage_static_params

    @staticmethod
    def _compute_a(input_data: InputData, b: float) -> float:
        """
        Calculates the coefficient 'a' based on manufacturer benchmarks.
        Derived from: N_rated = a * (DoD_rated)^(-b)

        Returns:
            float: The solved coefficient 'a'.
        """
        sp = input_data.storage_static_params
        n_rated = sp.degradation.n_cycles
        dod_rated = sp.deg_model.reference_dod

        return n_rated * (dod_rated ** b)

    def degradation(
            self,
            cycle_t: np.ndarray,
            temperature_t: np.ndarray,
    ) -> float:
        """
        Calculates the cumulative capacity loss for a given time-series period.

        Args:
            cycle_t: Time-series of State of Charge [0, 1] coefficients.
            temperature_t: Time-series of internal cell temperatures [K].

        Returns:
            float: Total fractional capacity loss increment (e.g., 0.0001).
        """
        assert len(cycle_t) == len(temperature_t)
        assert np.all(cycle_t <= 1.0) and np.all(cycle_t >= 0.0)

        activation_energy = self.sp.deg_model.activation_energy
        reference_temperature = self.sp.deg_model.reference_temperature

        # Precompute Arrhenius factor (time-local stress)
        # Ratio of reaction rate at current T vs reference T
        arrhenius_t = np.exp(
            (activation_energy / C.R) * (1 / reference_temperature - 1 / temperature_t)
        )

        return sum(
            self._compute_cycle_damage(cp, arrhenius_t)
            for cp in self._get_cycle_stats(cycle_t)
            if cp.dod > 1e-5
        )

    def _get_cycle_stats(self, cycle_t: np.ndarray) -> list[CycleParameters]:
        """
        Applies the Rainflow-counting algorithm to extract discrete cycles 
        from a continuous SOC profile.

        Args:
            cycle_t: Array of SOC coefficients [0, 1].

        Returns:
            list[CycleParameters]: A list of objects detailing every detected cycle.
        """
        # rainflow.extract_cycles returns (amplitude, mean, count, start_idx, end_idx)
        cycles_raw = rainflow.extract_cycles(cycle_t)
        soc_lim = self.sp.technical.soc_limits
        capacity_mwh = self.sp.technical.capacity

        result = []

        for amp, mean, count, start_idx, end_idx in cycles_raw:
            # Amplitude is (max-min)/2, so DoD (delta) is 2 * amplitude
            # We multiply by capacity to get the DoD in [MWh]
            dod_mwh = 2 * amp * capacity_mwh

            # Normalize the average SOC of the cycle to the allowed operational window
            soc_avg_raw = mean
            soc_avg = (soc_avg_raw - soc_lim.min) / (soc_lim.max - soc_lim.min)
            soc_avg = np.clip(soc_avg, 0.0, 1.0)

            result.append(
                CycleParameters(
                    dod=dod_mwh,
                    soc_avg=soc_avg,
                    count=count,
                    start_idx=start_idx,
                    end_idx=end_idx,
                )
            )

        return result

    def _compute_cycle_damage(
            self,
            cp: CycleParameters,
            arrhenius_t: np.ndarray,
    ) -> float:
        """
        Calculates the damage for one specific cycle using the power-law life model.

        Args:
            cp: Extracted cycle characteristics.
            arrhenius_t: Array of precomputed thermal stress factors.

        Returns:
            float: Incremental capacity loss (fraction of total capacity).
        """
        q_init = self.sp.technical.capacity

        # Convert MWh swing back to a fraction [0, 1] relative to nameplate capacity
        dod_fraction = cp.dod / q_init

        # n_base: The total number of cycles the battery could survive 
        # at this specific DoD amplitude under reference temperature.
        # Calculated as: a * (DoD)^-b
        n_base = self._a * (dod_fraction ** -self._cycle_severity_exponent)

        # Average the thermal stress factor across the indices where this cycle occurred
        temp_factor = arrhenius_t[cp.start_idx:cp.end_idx + 1].mean()

        # Incremental Damage = (Number of cycles / Total allowable cycles) * Thermal Stress
        damage = cp.count * temp_factor / n_base

        return damage