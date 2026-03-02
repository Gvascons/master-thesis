"""XGBoost model wrapper."""

import xgboost as xgb

from src.models.base import BaseModel


class XGBoostModel(BaseModel):
    MODEL_NAME = "xgboost"
    FAMILY = "gbdt"
    SUPPORTS_GPU = True

    def __init__(self, task_type: str, n_classes: int | None = None, seed: int = 42, **kwargs):
        # Remove cat_feature_indices (used by CatBoost only) before passing to XGBoost
        kwargs.pop("cat_feature_indices", None)

        super().__init__(task_type, n_classes, seed=seed, **kwargs)

        # Set objective based on task
        if task_type == "binary":
            kwargs.setdefault("objective", "binary:logistic")
            kwargs.setdefault("eval_metric", "auc")
        elif task_type == "multiclass":
            kwargs.setdefault("objective", "multi:softprob")
            kwargs.setdefault("eval_metric", "mlogloss")
            kwargs["num_class"] = n_classes
        else:
            kwargs.setdefault("objective", "reg:squarederror")
            kwargs.setdefault("eval_metric", "rmse")

        kwargs.setdefault("n_estimators", 1000)
        kwargs.setdefault("random_state", self.seed)
        kwargs.setdefault("n_jobs", -1)
        kwargs.setdefault("verbosity", 0)
        self.early_stopping_rounds = kwargs.pop("early_stopping_rounds", 50)

        self.params = kwargs
        # Note: early_stopping_rounds is set as a constructor param (XGBoost >=2.0)
        # but only takes effect when eval_set is provided in fit().
        cls = xgb.XGBClassifier if task_type != "regression" else xgb.XGBRegressor
        self.model = cls(early_stopping_rounds=self.early_stopping_rounds, **kwargs)

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        fit_kwargs = {}
        if X_val is not None and y_val is not None:
            fit_kwargs["eval_set"] = [(X_val, y_val)]
            fit_kwargs["verbose"] = False
        else:
            # Disable early stopping when no validation set is available
            self.model.set_params(early_stopping_rounds=None)
        self.model.fit(X_train, y_train, **fit_kwargs)
        self.is_fitted = True
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        if self.task_type == "regression":
            raise NotImplementedError("predict_proba not available for regression")
        return self.model.predict_proba(X)
