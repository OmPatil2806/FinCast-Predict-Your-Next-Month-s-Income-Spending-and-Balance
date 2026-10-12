"""Small shared helpers: project root, config loading, paths and random seeds."""

import random
from functools import lru_cache
from pathlib import Path

import numpy as np
import yaml


def get_project_root() -> Path:
    """Return the repository root (the folder that holds config/ and src/)."""
    # this file lives at <root>/src/utils/helpers.py
    return Path(__file__).resolve().parents[2]


@lru_cache(maxsize=None)
def load_config(config_path: str = "config/config.yaml") -> dict:
    """Read the YAML settings once and return them as a dictionary."""
    with open(get_project_root() / config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(relative_path: str) -> Path:
    """Turn a path from config.yaml (relative to the project root) into a full path."""
    return get_project_root() / relative_path


def set_seed(seed: int | None = None) -> int:
    """Fix the random generators so results can be reproduced. Defaults to the config seed."""
    if seed is None:
        seed = load_config()["project"]["seed"]
    random.seed(seed)
    np.random.seed(seed)
    return seed
