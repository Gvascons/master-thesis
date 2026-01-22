"""Abstract base model interface."""

from abc import ABC, abstractmethod

import numpy as np


class BaseModel(ABC):
    """Abstract interface that all models must implement."""

    # Override in subclass
    MODEL_NAME: str = "base"
    FAMILY: str = "unknown"  # "gbdt", "deep_learning", "foundation_model"
    SUPPORTS_GPU: bool = False

    def __init__(self, task_type: str, n_classes: int | None = None, seed: int = 42, **kwargs):
        """
        Args:
            task_type: "binary", "multiclass", or "regression"
            n_classes: Number of classes (for classification)
            seed: Random seed for reproducibility
            **kwargs: Model-specific hyperparameters
        """
        self.task_type = task_type
        self.n_classes = n_classes
        self.seed = seed
        self.params = kwargs
        self.model = None
        self.is_fitted = False

    @abstractmethod
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> "BaseModel":
        """Train the model. Returns self for chaining."""
        ...

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return class labels (classification) or continuous values (regression)."""
        ...

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return probability estimates (classification only).

        Returns array of shape (n_samples, n_classes).
        For regression models, this may raise NotImplementedError.
        """
        ...

    def get_params(self) -> dict:
        """Return current hyperparameters."""
        return self.params.copy()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(task={self.task_type}, params={self.params})"
