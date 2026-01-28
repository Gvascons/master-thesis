"""TabPFN v2.5 model wrapper (tabpfn>=6.0).

TabPFN v2.5 natively handles datasets up to 50K samples and 2K features.
For datasets exceeding 50K, a subsampling ensemble is used as fallback.
"""

import logging

import numpy as np

from src.models.base import BaseModel

logger = logging.getLogger("tabular_benchmark")


class TabPFNModel(BaseModel):
    MODEL_NAME = "tabpfn"
    FAMILY = "foundation_model"
    SUPPORTS_GPU = True

    # v2.5 native limit (up from 10K in v2)
    V25_MAX_SAMPLES = 50_000

    def __init__(self, task_type: str, n_classes: int | None = None, seed: int = 42, **kwargs):
        super().__init__(task_type, n_classes, seed=seed, **kwargs)
        self.max_samples = kwargs.pop("max_samples", self.V25_MAX_SAMPLES)
        self.n_ensemble_configs = kwargs.pop("n_ensemble_configs", 4)
        self.n_estimators = kwargs.pop("n_estimators", 8)
        self._ensemble_models = []

    def _create_tabpfn(self):
        """Create a TabPFN v2.5 instance."""
        from tabpfn import TabPFNClassifier, TabPFNRegressor

        if self.task_type in ("binary", "multiclass"):
            return TabPFNClassifier(
                device="auto",
                n_estimators=self.n_estimators,
                random_state=self.seed,
            )
        else:
            return TabPFNRegressor(
                device="auto",
                n_estimators=self.n_estimators,
                random_state=self.seed,
            )

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """Fit TabPFN v2.5. Zero-shot but needs to store training data.

        v2.5 handles up to 50K samples natively. For larger datasets,
        a subsampling ensemble is used as fallback.
        """
        n_samples = X_train.shape[0]
        self._ensemble_models = []

        # Convert to numpy for safe indexing
        X_arr = X_train.values if hasattr(X_train, "values") else X_train

        if n_samples <= self.max_samples:
            # Direct fit — v2.5 handles up to 50K natively
            model = self._create_tabpfn()
            model.fit(X_arr, y_train)
            self._ensemble_models.append(model)
            logger.info(f"TabPFN v2.5: single model, {n_samples} samples")
        else:
            # Subsampling ensemble for datasets exceeding v2.5 native limit
            rng = np.random.RandomState(self.seed)
            for i in range(self.n_ensemble_configs):
                idx = rng.choice(n_samples, size=self.max_samples, replace=False)
                model = self._create_tabpfn()
                model.fit(X_arr[idx], y_train[idx])
                self._ensemble_models.append(model)
            logger.info(
                f"TabPFN v2.5: {self.n_ensemble_configs}-member ensemble, "
                f"{n_samples} samples subsampled to {self.max_samples}"
            )

        self.is_fitted = True
        return self

    def predict(self, X):
        if self.task_type in ("binary", "multiclass"):
            proba = self.predict_proba(X)
            return np.argmax(proba, axis=1)
        else:
            # Average regression predictions
            preds = [m.predict(X) for m in self._ensemble_models]
            return np.mean(preds, axis=0)

    def predict_proba(self, X):
        if self.task_type == "regression":
            raise NotImplementedError("predict_proba not available for regression")
        # Average probabilities across ensemble
        proba_list = [m.predict_proba(X) for m in self._ensemble_models]
        return np.mean(proba_list, axis=0)
