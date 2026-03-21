from src.preprocess.config import Configuration
from src.preprocess.load.market import load_market_data
from src.preprocess.load.params import load_params
from src.preprocess.load.storage import load_storage_static_params, load_storage_state_params
from src.preprocess.model import InputData


def load_input_data(configuration: Configuration) -> InputData:
    return InputData(
        market_data=load_market_data(configuration.input_paths.market_data),
        storage_static_params=load_storage_static_params(configuration.input_paths.storage_static_params),
        storage_state_params=load_storage_state_params(configuration.input_paths.storage_state_params),
        params=load_params(configuration.input_paths.params),
    )