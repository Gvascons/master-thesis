"""Configuration loader using OmegaConf."""

from pathlib import Path

from omegaconf import DictConfig, OmegaConf

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_CONFIGS_DIR = _PROJECT_ROOT / "configs"


def load_config(name: str) -> DictConfig:
    """Load a YAML config file by name (without extension)."""
    path = _CONFIGS_DIR / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    return OmegaConf.load(path)


def load_datasets_config() -> DictConfig:
    return load_config("datasets")


def load_models_config() -> DictConfig:
    return load_config("models")


def load_experiment_config() -> DictConfig:
    return load_config("experiment")


def get_project_root() -> Path:
    return _PROJECT_ROOT
