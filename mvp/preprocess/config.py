import dataclasses
from pathlib import Path
from typing import Self
from configparser import ConfigParser


class ConfigValidationError(Exception):
    pass


def validate_file_suffix_path(path: Path, param_name: str, suffix: str) -> None:
    if not path.exists():
        raise ConfigValidationError(f"given path {path} does not exist")
    if not path.is_file():
        raise ConfigValidationError(f"path specified as {param_name}: {path} is not a file")
    if not path.suffix == suffix:
        raise ConfigValidationError(f"path specified as {param_name}: {path} is not a *{suffix} file")


def validate_directory_path(path: Path, param_name: str) -> None:
    if not path.exists():
        raise ConfigValidationError(f"given path {path} does not exist")
    if not path.is_dir():
        raise ConfigValidationError(f"path specified as {param_name}: {path} is not a directory")


@dataclasses.dataclass
class InputPaths:
    """Config section containing input data paths"""
    market_data_path: Path
    storage_params_path: Path

    @classmethod
    def load(cls, config: ConfigParser) -> Self:
        section = 'input'
        return cls(
            market_data_path=Path(config.get(section, 'market_data_path')),
            storage_params_path=Path(config.get(section, 'storage_params_path')),
        )

    def __post_init__(self):
        validate_file_suffix_path(
            self.market_data_path,
            param_name='market_data_path',
            suffix='.csv'
        )
        validate_file_suffix_path(
            self.storage_params_path,
            param_name='storage_params_path',
            suffix='.yaml'
        )



@dataclasses.dataclass
class MetaParameters:
    """Config section containing meta-parameters"""
    pass


@dataclasses.dataclass
class OutputPaths:
    """Config section containing output data paths"""
    result_dir: Path

    @classmethod
    def load(cls, config: ConfigParser) -> Self:
        section = 'output'
        return cls(
            result_dir=Path(config.get(section, 'result_dir')),
        )

    def __post_init__(self):
        validate_directory_path(self.result_dir, param_name='result_dir')


@dataclasses.dataclass
class Configuration:
    """Loaded configuration *.ini file"""
    input_paths: InputPaths
    output_paths: OutputPaths

    @classmethod
    def load(cls, path: Path) -> Self:
        validate_file_suffix_path(path, 'market_data_path', suffix='.ini')
        config_parser = ConfigParser()
        config_parser.read(path)
        return cls(
            input_paths=InputPaths.load(config_parser),
            output_paths=OutputPaths.load(config_parser),
        )
