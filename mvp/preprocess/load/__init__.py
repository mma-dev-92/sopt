from mvp.preprocess.config import Configuration
from mvp.preprocess.load.market import load_market_data
from mvp.preprocess.load.storage import load_storage_params
from mvp.preprocess.model import InputData


def load_input_data(configuration: Configuration) -> InputData:
    return InputData(
        market_data=load_market_data(configuration.input_paths.market_data_path),
        storage_params=load_storage_params(configuration.input_paths.storage_params_path),
    )