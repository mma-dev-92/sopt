import pandas as pd

from src.cli.run import run_opt
from src.preprocess.model import StorageStaticParams, InputData


class MultiOptRunner:
    """
    Multiple optimization problems runner

    Used to run multiple pipelines, gather the results and summarize them. It is very simplified, assumptions:
        * revenue degradation_model = storage parameters degradation_model
        * running optimization for each day in given year (assuming perfect forecast, using DA market data)

    Aim of the experiment: to access if the idea is not terribly wrong.
    """

    def __init__(self, input_data: InputData) -> None:
        self.input_data = input_data

    def compute_yearly_revenue(self, year: int, save_files: bool = False) -> float:
        rev = 0.0
        partitions = pd.date_range(
            start=pd.Timestamp(year=year, month=1, day=1),
            stop=pd.Timestamp(year=year, month=12, day=31),
        )
        for partition in partitions:
            self.input_data.params.partition = partition
            results_df = run_opt(self.input_data)
            rev += results_df['rev'].sum()

        return rev

    def compute_npv(
            self,
            base_y_rev: float,
            wacc_range: list[float],
            deg_rate: float
    ) -> pd.Series:
        return pd.Series({
            wacc: self._compute_single_npv(self.input_data.storage_static_params, base_y_rev, wacc, deg_rate)
            for wacc in wacc_range
        })

    def compute_irr(
            self,
            results: list[pd.DataFrame],
            deg_rate_range: list[float]
    ) -> pd.Series:
        pass

    @staticmethod
    def _compute_single_npv(
            storage_params: StorageStaticParams,
            base_y_rev: float,
            wacc: float,
            deg_rate: float
    ) -> float:

        capex = storage_params.capex
        opex = storage_params.opex
        capacity = storage_params.capacity
        life_time = storage_params.lifetime_years

        total_capex = capacity * capex

        npv = -total_capex

        for y in range(1, life_time + 1):
            discount = (1 + wacc) ** (-y)
            degradation = (1 - deg_rate) ** (y - 1)

            revenue = base_y_rev * degradation
            yearly_opex = capacity * opex

            npv += discount * (revenue - yearly_opex)

        return npv

    # TODO: bin search would be nice, with init left=0, right=1.0
    @staticmethod
    def _compute_single_irr(storage_params: StorageStaticParams, deg_rate: float) -> float:
        pass
