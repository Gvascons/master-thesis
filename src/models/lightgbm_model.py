"""LightGBM model wrapper."""

import lightgbm as lgb

from src.models.base import BaseModel


class LightGBMModel(BaseModel):
    MODEL_NAME = "lightgbm"
    FAMILY = "gbdt"
    SUPPORTS_GPU = True

    def __init__(self, task_type: str, n_classes: int | None = None, seed: int = 42, **kwargs):
        # Remove cat_feature_indices (used by CatBoost only) before passing to LightGBM
        kwargs.pop("cat_feature_indices", None)

        super().__init__(task_type, n_classes, seed=seed, **kwargs)

        if task_type == "binary":
            kwargs.setdefault("objective", "binary")
            kwargs.setdefault("metric", "auc")
        elif task_type == "multiclass":
            kwargs.setdefault("objective", "multiclass")
            kwargs.setdefault("metric", "multi_logloss")
            kwargs["num_class"] = n_classes
        else:
            kwargs.setdefault("objective", "regression")
            kwargs.setdefault("metric", "rmse")

        kwargs.setdefault("n_estimators", 1000)
        kwargs.setdefault("random_state", self.seed)
        kwargs.setdefault("n_jobs", -1)
        kwargs.setdefault("verbosity", -1)

        self.params = kwargs
        if task_type != "regression":
            self.model = lgb.LGBMClassifier(**kwargs)
        else:
            self.model = lgb.LGBMRegressor(**kwargs)

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        fit_kwargs = {}
        if X_val is not None and y_val is not None:
            fit_kwargs["eval_set"] = [(X_val, y_val)]
            fit_kwargs["callbacks"] = [
                lgb.log_evaluation(period=0),
                lgb.early_stopping(50, verbose=False),
            ]
        self.model.fit(X_train, y_train, **fit_kwargs)
        self.is_fitted = True
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        if self.task_type == "regression":
            raise NotImplementedError("predict_proba not available for regression")
        return self.model.predict_proba(X)
