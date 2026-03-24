import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

load_dotenv()

_CONFIG_DIR = Path(__file__).parent.parent / "configs"


class ConfigLoader:
    """Loads and merges YAML configuration files."""

    _cache: dict[str, dict] = {}

    @classmethod
    def load(cls, config_name: str) -> dict:
        """Load a YAML config file by name (without .yaml extension)."""
        if config_name in cls._cache:
            return cls._cache[config_name]

        config_path = _CONFIG_DIR / f"{config_name}.yaml"
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        cls._cache[config_name] = config
        return config

    @classmethod
    def get(cls, config_name: str, key: str, default: Any = None) -> Any:
        """Get a nested key from a config using dot notation."""
        config = cls.load(config_name)
        keys = key.split(".")
        value = config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k, default)
            else:
                return default
        return value

    @classmethod
    def load_all(cls) -> dict:
        """Load all configs into a single merged dict."""
        all_configs = {}
        for yaml_file in _CONFIG_DIR.glob("*.yaml"):
            name = yaml_file.stem
            all_configs[name] = cls.load(name)
        return all_configs

    @classmethod
    def clear_cache(cls) -> None:
        cls._cache.clear()
