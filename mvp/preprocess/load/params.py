from pathlib import Path

import yaml

from mvp.preprocess.model.params import Params


def load_params(path: Path) -> Params:
    with open(path, 'rb') as f:
        data = yaml.safe_load(f)
        return Params(**data)
