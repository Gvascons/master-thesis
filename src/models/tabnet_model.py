"""TabNet model wrapper using pytorch-tabnet's sklearn-like API."""

import logging

import numpy as np
from pytorch_tabnet.tab_model import TabNetClassifier, TabNetRegressor

from src.models.base import BaseModel

logger = logging.getLogger("tabular_benchmark")


class TabNetModel(BaseModel):
    MODEL_NAME = "tabnet"
    FAMILY = "deep_learning"
    SUPPORTS_GPU = True

    def __init__(self, task_type: str, n_classes: int | None = None, seed: int = 42, **kwargs):
        super().__init__(task_type, n_classes, seed=seed, **kwargs)

        self.max_epochs = kwargs.pop("max_epochs", 200)
        self.patience = kwargs.pop("patience", 20)
        self.batch_size = kwargs.pop("batch_size", 256)
        self.lr = kwargs.pop("learning_rate", 0.02)

        tabnet_params = {
            "n_d": kwargs.pop("n_d", 16),
            "n_a": kwargs.pop("n_a", 16),
            "n_steps": kwargs.pop("n_steps", 5),
            "gamma": kwargs.pop("gamma", 1.5),
            "lambda_sparse": kwargs.pop("lambda_sparse", 1e-4),
            "optimizer_params": {"lr": self.lr},
            "scheduler_params": {"step_size": 50, "gamma": 0.9},
            "scheduler_fn": __import__("torch.optim.lr_scheduler", fromlist=["StepLR"]).StepLR,
            "mask_type": kwargs.pop("mask_type", "sparsemax"),
            "seed": self.seed,
            "verbose": 0,
        }

        if task_type != "regression":
            self.model = TabNetClassifier(**tabnet_params)
        else:
            self.model = TabNetRegressor(**tabnet_params)

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        fit_kwargs = {
            "max_epochs": self.max_epochs,
            "patience": self.patience,
            "batch_size": self.batch_size,
        }

        # TabNet requires float32 input
        X_tr = X_train.astype(np.float32) if X_train.dtype != np.float32 else X_train
        y_tr = y_train

        if self.task_type == "regression":
            y_tr = y_train.reshape(-1, 1).astype(np.float32)

        if X_val is not None and y_val is not None:
            X_va = X_val.astype(np.float32) if X_val.dtype != np.float32 else X_val
            y_va = y_val
            if self.task_type == "regression":
                y_va = y_val.reshape(-1, 1).astype(np.float32)
            fit_kwargs["eval_set"] = [(X_va, y_va)]
            if self.task_type == "binary":
                fit_kwargs["eval_metric"] = ["auc"]
            elif self.task_type == "multiclass":
                fit_kwargs["eval_metric"] = ["logloss"]
            else:
                fit_kwargs["eval_metric"] = ["rmse"]

        self.model.fit(X_tr, y_tr, **fit_kwargs)
        self.is_fitted = True
        return self

    def predict(self, X):
        X = X.astype(np.float32) if X.dtype != np.float32 else X
        preds = self.model.predict(X)
        if self.task_type == "regression":
            return preds.flatten()
        return preds

    def predict_proba(self, X):
        if self.task_type == "regression":
            raise NotImplementedError("predict_proba not available for regression")
        X = X.astype(np.float32) if X.dtype != np.float32 else X
        return self.model.predict_proba(X)
