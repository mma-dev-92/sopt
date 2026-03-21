from pathlib import Path

import yaml

from src.preprocess.model import StorageStaticParams
from src.preprocess.model.storage import StorageStateParams


def load_storage_static_params(path: Path) -> StorageStaticParams:
    with open(path, 'rb') as f:
        data = yaml.safe_load(f)
        return StorageStaticParams(**data['storage'])


def load_storage_state_params(path: Path) -> StorageStateParams:
    with open(path, 'rb') as f:
        data = yaml.safe_load(f)
        return StorageStateParams(**data['storage_state'])
