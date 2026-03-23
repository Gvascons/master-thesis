"""Capture runtime environment details for reproducibility."""

from __future__ import annotations

import platform
import sys
from importlib.metadata import version as pkg_version


def capture_environment() -> dict:
    """Return a dict describing the current runtime environment."""
    env: dict = {
        "python_version": sys.version,
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
    }

    # PyTorch / CUDA
    try:
        import torch

        env["torch_version"] = torch.__version__
        env["cuda_available"] = torch.cuda.is_available()
        env["cuda_version"] = torch.version.cuda
        if torch.cuda.is_available():
            env["gpu_name"] = torch.cuda.get_device_name(0)
            env["cudnn_version"] = torch.backends.cudnn.version()
    except ImportError:
        env["torch_version"] = None

    # Key package versions
    packages = [
        "numpy",
        "pandas",
        "scikit-learn",
        "xgboost",
        "lightgbm",
        "catboost",
        "optuna",
        "openml",
        "scipy",
        "statsmodels",
        "scikit-posthocs",
        "tabpfn",
        "tabm",
        "pytorch_tabnet",
        "rtdl_revisiting_models",
        "pytabkit",
        "einops",
    ]
    pkg_versions: dict[str, str | None] = {}
    for pkg in packages:
        try:
            pkg_versions[pkg] = pkg_version(pkg)
        except Exception:
            pkg_versions[pkg] = None
    env["packages"] = pkg_versions

    return env
