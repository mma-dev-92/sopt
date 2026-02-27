from mvp.preprocess.config import Configuration
from mvp.preprocess.load.market import load
from mvp.preprocess.load.storage import load
from mvp.preprocess.model import InputData


def load_input_data(configuration: Configuration) -> InputData:
    return InputData(
        market_data=load(configuration.input_paths.market_data_path),
        storage_params=load(configuration.input_paths.storage_params_path),
    )