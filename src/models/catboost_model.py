"""CatBoost model wrapper."""

import numpy as np
from catboost import CatBoostClassifier, CatBoostRegressor

from src.models.base import BaseModel


class CatBoostModel(BaseModel):
    MODEL_NAME = "catboost"
    FAMILY = "gbdt"
    SUPPORTS_GPU = True

    def __init__(self, task_type: str, n_classes: int | None = None, seed: int = 42, **kwargs):
        # Extract cat_feature_indices before passing kwargs to parent/CatBoost
        self._cat_feature_indices = kwargs.pop("cat_feature_indices", None)

        super().__init__(task_type, n_classes, seed=seed, **kwargs)

        kwargs.setdefault("iterations", 1000)
        kwargs.setdefault("random_seed", self.seed)
        kwargs.setdefault("verbose", 0)
        kwargs.setdefault("allow_writing_files", False)

        if task_type == "binary":
            kwargs.setdefault("loss_function", "Logloss")
            kwargs.setdefault("eval_metric", "AUC")
        elif task_type == "multiclass":
            kwargs.setdefault("loss_function", "MultiClass")
            kwargs.setdefault("eval_metric", "MultiClass")
        else:
            kwargs.setdefault("loss_function", "RMSE")
            kwargs.setdefault("eval_metric", "RMSE")

        self.params = kwargs
        if task_type != "regression":
            self.model = CatBoostClassifier(**kwargs)
        else:
            self.model = CatBoostRegressor(**kwargs)

    @staticmethod
    def _cast_cat_columns(X: np.ndarray, cat_indices: list[int]) -> np.ndarray:
        """Cast categorical columns from float to int for CatBoost compatibility.

        OrdinalEncoder outputs float64, but CatBoost rejects float arrays when
        cat_features is specified. Convert to object array with int cat columns.
        """
        X = X.copy().astype(object)
        for i in cat_indices:
            X[:, i] = X[:, i].astype(int)
        return X

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        fit_kwargs = {}
        if self._cat_feature_indices:
            X_train = self._cast_cat_columns(X_train, self._cat_feature_indices)
            fit_kwargs["cat_features"] = self._cat_feature_indices
        if X_val is not None and y_val is not None:
            if self._cat_feature_indices:
                X_val = self._cast_cat_columns(X_val, self._cat_feature_indices)
            fit_kwargs["eval_set"] = (X_val, y_val)
            fit_kwargs["early_stopping_rounds"] = 50
        self.model.fit(X_train, y_train, **fit_kwargs)
        self.is_fitted = True
        return self

    def _prepare_X(self, X: np.ndarray) -> np.ndarray:
        """Cast cat columns for prediction if needed."""
        if self._cat_feature_indices:
            return self._cast_cat_columns(X, self._cat_feature_indices)
        return X

    def predict(self, X):
        X = self._prepare_X(X)
        if self.task_type == "regression":
            return self.model.predict(X).flatten()
        return np.argmax(self.model.predict_proba(X), axis=1)

    def predict_proba(self, X):
        X = self._prepare_X(X)
        if self.task_type == "regression":
            raise NotImplementedError("predict_proba not available for regression")
        return self.model.predict_proba(X)
