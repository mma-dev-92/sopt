import dataclasses
from enum import Enum
from pathlib import Path
from typing import Self
from configparser import ConfigParser


class OutputFormat(Enum):
    CSV = "csv"
    XLSX = "xlsx"
    PARQUET = "parquet"


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
    market_data: Path
    storage_static_params: Path
    storage_state_params: Path
    params: Path

    @classmethod
    def load(cls, config: ConfigParser) -> Self:
        section = 'input'
        return cls(
            market_data=Path(config.get(section, 'market_data')),
            storage_static_params=Path(config.get(section, 'storage_static_params')),
            storage_state_params=Path(config.get(section, 'storage_state_params')),
            params=Path(config.get(section, 'params')),
        )

    def __post_init__(self):
        validate_file_suffix_path(
            self.market_data,
            param_name='market_data',
            suffix='.csv',
        )
        validate_file_suffix_path(
            self.storage_static_params,
            param_name='storage_static_params',
            suffix='.yaml',
        )
        validate_file_suffix_path(
            self.storage_static_params,
            param_name='storage_state_params',
            suffix='.yaml',
        )
        validate_file_suffix_path(
            self.params,
            param_name='params',
            suffix='.yaml',
        )


@dataclasses.dataclass
class OutputPaths:
    """Config section containing output data paths"""
    result_dir: Path
    format: str

    @classmethod
    def load(cls, config: ConfigParser) -> Self:
        section = 'output'
        return cls(
            result_dir=Path(config.get(section, 'result_dir')),
            format=config.get(section, 'format'),
        )

    def __post_init__(self):
        validate_directory_path(self.result_dir, param_name='result_dir')
        self.validate_output_format()

    def validate_output_format(self) -> None:
        if not self.format in OutputFormat:
            raise ConfigValidationError(
                f"format in [output] section should be one of {[x.value for x in OutputFormat]}, "
                f"but {self.format} was given"
            )


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
