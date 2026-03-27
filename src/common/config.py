from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv


def load_environment() -> None:
    load_dotenv()


def load_yaml_config(config_path: str | Path) -> dict[str, Any]:
    """
    Takes path to YAML file and returns YAML content parsed as dictionary
    """
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