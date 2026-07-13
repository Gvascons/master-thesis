"""TabFM (Google Research, 2026) model wrapper.

Zero-shot in-context foundation model of the same PFN/ICL family as TabPFN
(prior-fitted on synthetic SCM-generated datasets), released 2026-06-30 with
PyTorch weights on Hugging Face (google/tabfm-1.0.0-pytorch; ~13 GB across
separate classification and regression checkpoints). No peer-reviewed paper
as of 2026-07; cite the Google Research blog + model card. Version pinned in
pyproject.toml (git commit d8678b6).

Benchmark policies mirror the TabPFN wrapper exactly, for comparability
between the two foundation models:
  * rows capped at 50K (`max_num_rows`, native lib parameter — internal
    subsampling, same threshold as `tabpfn_max_samples`);
  * hard architectural limit of 10 classes — `helena` (100 classes) is out,
    as it already is for TabPFN;
  * library-default inference ensemble (n_estimators=32);
  * chunked prediction to bound GPU memory.
Weights run in bfloat16 (the release's compute default; ~3-7 GB VRAM).
"""

import logging

import numpy as np

from src.models.base import BaseModel

logger = logging.getLogger("tabular_benchmark")


class TabFMModel(BaseModel):
    MODEL_NAME = "tabfm"
    FAMILY = "foundation_model"
    SUPPORTS_GPU = True

    MAX_CLASSES = 10  # architectural limit (max_classes in the HF config)
    DEFAULT_MAX_ROWS = 50_000  # mirrors tabpfn_max_samples

    def __init__(self, task_type: str, n_classes: int | None = None, seed: int = 42, **kwargs):
        super().__init__(task_type, n_classes, seed=seed, **kwargs)
        if task_type in ("binary", "multiclass") and n_classes and n_classes > self.MAX_CLASSES:
            raise ValueError(
                f"TabFM supports at most {self.MAX_CLASSES} classes, got {n_classes}"
            )
        self.max_samples = kwargs.pop("max_samples", self.DEFAULT_MAX_ROWS)
        # 8, not the lib default of 32: the pilot measured identical quality
        # (adult AUC 0.9322 vs 0.9321) at 1/4 the inference cost, and 8 also
        # matches this benchmark's TabPFN ensemble policy.
        self.n_estimators = kwargs.pop("n_estimators", 8)
        self._proba_cache = None

    def _load_backbone(self):
        """Load the pretrained backbone (process-wide cached by the lib)."""
        import torch
        from tabfm import tabfm_v1_0_0_pytorch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_type = "regression" if self.task_type == "regression" else "classification"
        return tabfm_v1_0_0_pytorch.load(model_type, device=device)

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """Zero-shot: stores the (row-capped) training context; no gradient steps.

        X_val/y_val are unused, as with TabPFN.
        """
        from tabfm import TabFMClassifier, TabFMRegressor

        backbone = self._load_backbone()
        common = dict(
            model=backbone,
            n_estimators=self.n_estimators,
            max_num_rows=self.max_samples,
            random_state=self.seed,
        )
        if self.task_type in ("binary", "multiclass"):
            self.model = TabFMClassifier(**common)
        else:
            self.model = TabFMRegressor(**common)

        X_arr = X_train.values if hasattr(X_train, "values") else X_train
        self.model.fit(X_arr, y_train)
        n = X_arr.shape[0]
        logger.info(
            f"TabFM: context of {min(n, self.max_samples)}/{n} rows, "
            f"n_estimators={self.n_estimators}"
        )
        self.is_fitted = True
        return self

    # Same chunking rationale as the TabPFN wrapper: predictions are
    # independent across rows, so chunking bounds peak GPU memory.
    _PREDICT_BATCH = 2000

    def _batched(self, fn, X):
        X_arr = X.values if hasattr(X, "values") else X
        n = X_arr.shape[0]
        if n <= self._PREDICT_BATCH:
            return fn(X_arr)
        chunks = [fn(X_arr[i:i + self._PREDICT_BATCH]) for i in range(0, n, self._PREDICT_BATCH)]
        return np.concatenate(chunks, axis=0)

    def predict(self, X):
        if self.task_type in ("binary", "multiclass"):
            proba = self.predict_proba(X)
            return np.argmax(proba, axis=1)
        return self._batched(self.model.predict, X)

    def predict_proba(self, X):
        if self.task_type == "regression":
            raise NotImplementedError("predict_proba not available for regression")
        # Inference costs ~100ms/row at 35K context; metric computation calls
        # predict() and predict_proba() on the same X, so cache the last
        # result to avoid paying for two identical full passes.
        X_arr = X.values if hasattr(X, "values") else X
        if self._proba_cache is not None:
            cached_X, cached_proba = self._proba_cache
            if cached_X.shape == X_arr.shape and np.array_equal(cached_X, X_arr):
                return cached_proba
        proba = self._batched(self.model.predict_proba, X_arr)
        self._proba_cache = (X_arr.copy(), proba)
        return proba
