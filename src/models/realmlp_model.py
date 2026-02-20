"""RealMLP model wrapper using the ``pytabkit`` package.

Based on "Better by Default: Strong Pre-Tuned MLPs and Boosted Trees on Tabular Data"
(Holzmuller, Grinsztajn & Steinwart, NeurIPS 2024).

RealMLP improves on standard MLPs through:
  - Neural tangent parameterization for stable training
  - Parametric Mish activation functions
  - Specialized scaling layers (front scale)
  - Robust preprocessing (smooth clipping, robust scaling)
  - Piecewise-linear numerical embeddings

The wrapper delegates to ``pytabkit.RealMLP_TD_Classifier`` /
``pytabkit.RealMLP_TD_Regressor``, which provide a scikit-learn-compatible API.
"""

import logging

import numpy as np

from src.models.base import BaseModel

logger = logging.getLogger("tabular_benchmark")


class RealMLPModel(BaseModel):
    MODEL_NAME = "realmlp"
    FAMILY = "deep_learning"
    SUPPORTS_GPU = True

    def __init__(self, task_type: str, n_classes: int | None = None, seed: int = 42, **kwargs):
        super().__init__(task_type, n_classes, seed=seed, **kwargs)

        # Training params
        self.n_epochs = kwargs.pop("n_epochs", 256)
        self.batch_size = kwargs.pop("batch_size", 256)
        self.lr = kwargs.pop("learning_rate", 0.04)
        self.wd = kwargs.pop("weight_decay", 0.0)
        self.p_drop = kwargs.pop("dropout", 0.15)

        # Architecture params
        self.n_hidden_layers = kwargs.pop("n_hidden_layers", 3)
        self.hidden_width = kwargs.pop("hidden_width", 256)
        self.act = kwargs.pop("activation", "mish")
        self.use_parametric_act = kwargs.pop("use_parametric_act", True)
        self.add_front_scale = kwargs.pop("add_front_scale", True)

        # Numerical embedding params
        self.num_emb_type = kwargs.pop("num_emb_type", "pbld")

        # Label smoothing
        self.ls_eps = kwargs.pop("ls_eps", 0.1)

        # Early stopping
        self.use_early_stopping = kwargs.pop("use_early_stopping", True)
        self.patience = kwargs.pop("patience", 40)

    def _create_model(self):
        """Create the pytabkit RealMLP model."""
        from pytabkit import RealMLP_TD_Classifier, RealMLP_TD_Regressor

        hidden_sizes = [self.hidden_width] * self.n_hidden_layers

        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"

        common_kwargs = dict(
            device=device,
            random_state=self.seed,
            n_cv=1,
            n_refit=0,
            n_epochs=self.n_epochs,
            batch_size=self.batch_size,
            hidden_sizes=hidden_sizes,
            lr=self.lr,
            wd=self.wd,
            p_drop=self.p_drop,
            act=self.act,
            use_parametric_act=self.use_parametric_act,
            add_front_scale=self.add_front_scale,
            num_emb_type=self.num_emb_type,
            use_early_stopping=self.use_early_stopping,
            early_stopping_additive_patience=self.patience,
            verbosity=0,
        )

        if self.task_type in ("binary", "multiclass"):
            return RealMLP_TD_Classifier(
                **common_kwargs,
                use_ls=self.ls_eps > 0,
                ls_eps=self.ls_eps,
            )
        else:
            return RealMLP_TD_Regressor(**common_kwargs)

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        self.model = self._create_model()

        # pytabkit accepts numpy arrays and DataFrames
        X_arr = X_train if isinstance(X_train, np.ndarray) else np.asarray(X_train)
        y_arr = y_train if isinstance(y_train, np.ndarray) else np.asarray(y_train)

        fit_kwargs = {}
        if X_val is not None and y_val is not None:
            X_v = X_val if isinstance(X_val, np.ndarray) else np.asarray(X_val)
            y_v = y_val if isinstance(y_val, np.ndarray) else np.asarray(y_val)
            fit_kwargs["X_val"] = X_v
            fit_kwargs["y_val"] = y_v

        self.model.fit(X_arr, y_arr, **fit_kwargs)
        self.is_fitted = True
        return self

    def predict(self, X):
        X_arr = X if isinstance(X, np.ndarray) else np.asarray(X)
        return self.model.predict(X_arr)

    def predict_proba(self, X):
        if self.task_type == "regression":
            raise NotImplementedError("predict_proba not available for regression")
        X_arr = X if isinstance(X, np.ndarray) else np.asarray(X)
        return self.model.predict_proba(X_arr)
