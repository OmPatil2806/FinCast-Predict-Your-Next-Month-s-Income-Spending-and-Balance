import random

import numpy as np

from src.utils.helpers import get_project_root, load_config, resolve_path, set_seed


def test_project_root_contains_config():
    root = get_project_root()
    assert (root / "config" / "config.yaml").is_file()
    assert (root / "src").is_dir()


def test_load_config_returns_settings():
    config = load_config()
    assert config["project"]["seed"] == 42
    assert config["locale"] == {"country": "DE", "currency": "EUR"}
    assert config["forecast"]["quantiles"] == [0.1, 0.5, 0.9]


def test_resolve_path_is_inside_project():
    path = resolve_path(load_config()["paths"]["data_raw"])
    assert path.is_absolute()
    assert path == get_project_root() / "data" / "raw"


def test_set_seed_makes_results_repeatable():
    set_seed(42)
    first = (random.random(), np.random.rand())
    set_seed(42)
    second = (random.random(), np.random.rand())
    assert first == second
