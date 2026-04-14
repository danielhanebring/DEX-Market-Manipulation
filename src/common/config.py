from __future__ import annotations

import os
from pathlib import Path
from typing import Any

def load_environment() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return

    load_dotenv()


def load_yaml_config(config_path: str | Path) -> dict[str, Any]:
    """
    Takes path to YAML file and returns YAML content parsed as dictionary
    """
    try:
        import yaml 
    except ImportError as exc:
        raise ImportError(
            "PyYAML is required to load YAML config files"
            "or pip install -r requirements.txt"
        ) from exc

    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file does not exist: {path}")

    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError(f"Invalid or empty YAML config: {path}")

    return config


def get_required_env_var(variable_name: str) -> str:
    """
    Gets the value of a required enviroment variable
    """
    value = os.getenv(variable_name)
    if not value:
        raise EnvironmentError(
            f"Missing required environment variable: {variable_name}"
        )
    return value