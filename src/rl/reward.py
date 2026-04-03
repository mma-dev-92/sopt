

class RewardCalculator:
    """
    Decouples the financial metric from the simulation engine.
    """

    @staticmethod
    def calculate(revenue: float, capacity_loss_increment: float) -> float:
        """
        Currently returning pure revenue.
        The capacity_loss_increment is passed in case you later want to
        calculate a 'profit minus physical asset depreciation' metric.
        """
        return revenue