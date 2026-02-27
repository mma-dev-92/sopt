from pathlib import Path

import yaml

from mvp.preprocess.model import StorageParams


def load(path: Path) -> StorageParams:
    with open(path, 'rb') as f:
        data = yaml.safe_load(f)
        return StorageParams(**data)
