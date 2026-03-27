from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIGS_DIR = PROJECT_ROOT / "configs"
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
FEATURES_DIR = DATA_DIR / "features"
LABELS_DIR = DATA_DIR / "labels"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"


def ensure_directory(directory: str | Path) -> Path:
    """
    Checks if dir exists, otherwise create the dir
    """
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    return path