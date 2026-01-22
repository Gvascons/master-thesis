"""Model factory: name -> class registry."""

import logging

from src.models.base import BaseModel

logger = logging.getLogger("tabular_benchmark")

# Registry populated lazily to avoid import errors when optional deps are missing
_REGISTRY: dict[str, type[BaseModel]] = {}
_REGISTRY_POPULATED = False


def _populate_registry():
    """Import all model classes and register them.

    Each import is wrapped individually so that a missing optional dependency
    only disables that single model instead of breaking the entire registry.
    """
    global _REGISTRY_POPULATED
    if _REGISTRY_POPULATED:
        return

    _model_imports = [
        ("src.models.xgboost_model", "XGBoostModel"),
        ("src.models.lightgbm_model", "LightGBMModel"),
        ("src.models.catboost_model", "CatBoostModel"),
        ("src.models.ft_transformer", "FTTransformerModel"),
        ("src.models.tabnet_model", "TabNetModel"),
        ("src.models.saint_model", "SAINTModel"),
        ("src.models.tabpfn_model", "TabPFNModel"),
        ("src.models.tabm_model", "TabMModel"),
        ("src.models.realmlp_model", "RealMLPModel"),
        ("src.models.mlp_model", "MLPModel"),
    ]

    for module_path, class_name in _model_imports:
        try:
            import importlib
            module = importlib.import_module(module_path)
            cls = getattr(module, class_name)
            _REGISTRY[cls.MODEL_NAME] = cls
        except (ImportError, Exception) as e:
            logger.warning(f"Could not load {class_name} from {module_path}: {e}")

    _REGISTRY_POPULATED = True


def create_model(
    model_name: str,
    task_type: str,
    n_classes: int | None = None,
    seed: int = 42,
    **kwargs,
) -> BaseModel:
    """Create a model instance by name."""
    _populate_registry()
    if model_name not in _REGISTRY:
        raise ValueError(
            f"Unknown model: {model_name}. Available: {list(_REGISTRY.keys())}"
        )
    return _REGISTRY[model_name](task_type=task_type, n_classes=n_classes, seed=seed, **kwargs)


def get_model_class(model_name: str) -> type[BaseModel]:
    """Get the model class without instantiation."""
    _populate_registry()
    if model_name not in _REGISTRY:
        raise ValueError(f"Unknown model: {model_name}")
    return _REGISTRY[model_name]


def list_models() -> list[str]:
    """List all registered model names."""
    _populate_registry()
    return list(_REGISTRY.keys())


def get_model_family(model_name: str) -> str:
    """Get the family (gbdt, deep_learning, foundation_model) for a model."""
    cls = get_model_class(model_name)
    return cls.FAMILY
