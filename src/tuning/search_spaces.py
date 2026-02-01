"""Convert YAML search space definitions to Optuna trial.suggest_* calls."""

from omegaconf import DictConfig

import optuna

from src.utils.config import load_models_config


def suggest_params(trial: optuna.Trial, model_name: str) -> dict:
    """Sample hyperparameters from the YAML-defined search space.

    Returns a dict of {param_name: sampled_value} suitable for model __init__.
    """
    models_cfg = load_models_config()

    if model_name not in models_cfg:
        raise ValueError(f"No search space defined for '{model_name}'")

    space = models_cfg[model_name]
    params = {}

    for param_name, spec in space.items():
        if spec is None:
            continue
        ptype = spec["type"]

        if ptype == "int":
            params[param_name] = trial.suggest_int(
                param_name, spec["low"], spec["high"],
                log=spec.get("log", False),
            )
        elif ptype == "float":
            params[param_name] = trial.suggest_float(
                param_name, spec["low"], spec["high"],
                log=spec.get("log", False),
            )
        elif ptype == "categorical":
            params[param_name] = trial.suggest_categorical(
                param_name, spec["choices"],
            )
        else:
            raise ValueError(f"Unknown param type '{ptype}' for {model_name}.{param_name}")

    return params


def get_search_space_names() -> list[str]:
    """Return all model names that have search spaces defined."""
    cfg = load_models_config()
    return [name for name in cfg if cfg[name] is not None]
