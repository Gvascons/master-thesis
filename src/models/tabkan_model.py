"""TabKAN (Chebyshev-KAN variant) with custom PyTorch training loop.

Reference: Eslamian, Aghaei & Cheng, "TabKAN: Advancing Tabular Data Analysis
using Kolmogorov-Arnold Network" (arXiv 2504.06559; Springer MLCSE 2025).
Own reimplementation from the paper's equations — the official `tabkan`
package trains with full-batch L-BFGS, has no early stopping, no
predict_proba and no regression pipeline, so it cannot follow this
benchmark's protocol (same situation as SAINT/STab, which were also
reimplemented from their papers).

We fix the ChebyKAN variant — the paper's best on average — instead of
tuning over all basis variants, so the 25-trial Optuna budget stays
comparable with the other DL models. Each edge learns a degree-D Chebyshev
polynomial: inputs are mapped into [-1, 1] by tanh (the polynomials' domain),
T_0..T_D are built by the recurrence T_k = 2x·T_{k-1} − T_{k-2}, and the
layer output is the coefficient-weighted sum. LayerNorm before each layer
keeps pre-tanh values in the non-saturated range. Trained with AdamW +
mini-batches + early stopping (declared deviation from the paper's L-BFGS,
adopted for protocol comparability across all DL models).
"""

import logging
import math

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.base import BaseModel

logger = logging.getLogger("tabular_benchmark")


class _ChebyKANLayer(nn.Module):
    """Edge-wise learnable Chebyshev polynomial layer (TabKAN eq. 14-16)."""

    def __init__(self, d_in: int, d_out: int, degree: int):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.degree = degree
        self.coeffs = nn.Parameter(torch.empty(d_in, d_out, degree + 1))
        # variance-preserving init over the (d_in * (degree+1)) summed terms
        nn.init.normal_(self.coeffs, mean=0.0, std=1.0 / math.sqrt(d_in * (degree + 1)))

    def forward(self, x):
        x = torch.tanh(x)  # map into [-1, 1], the Chebyshev domain
        cheb = [torch.ones_like(x), x]
        for _ in range(2, self.degree + 1):
            cheb.append(2 * x * cheb[-1] - cheb[-2])
        basis = torch.stack(cheb[: self.degree + 1], dim=-1)  # (batch, d_in, D+1)
        return torch.einsum("bid,iod->bo", basis, self.coeffs)


class _TabKANNet(nn.Module):
    """[LayerNorm -> ChebyKANLayer] x n_layers -> LayerNorm -> head layer."""

    def __init__(self, n_features: int, d_hidden: int, n_layers: int,
                 degree: int, d_out: int):
        super().__init__()
        dims = [n_features] + [d_hidden] * n_layers + [d_out]
        self.norms = nn.ModuleList(nn.LayerNorm(d_in) for d_in in dims[:-1])
        self.kan_layers = nn.ModuleList(
            _ChebyKANLayer(d_in, d_out_, degree)
            for d_in, d_out_ in zip(dims[:-1], dims[1:])
        )

    def forward(self, x):
        for norm, layer in zip(self.norms, self.kan_layers):
            x = layer(norm(x))
        return x


class TabKANModel(BaseModel):
    MODEL_NAME = "tabkan"
    FAMILY = "deep_learning"
    SUPPORTS_GPU = True

    def __init__(self, task_type: str, n_classes: int | None = None, seed: int = 42, **kwargs):
        super().__init__(task_type, n_classes, seed=seed, **kwargs)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_epochs = kwargs.pop("max_epochs", 200)
        self.patience = kwargs.pop("patience", 20)
        self.batch_size = kwargs.pop("batch_size", 256)
        self.lr = kwargs.pop("learning_rate", 1e-3)
        self.weight_decay = kwargs.pop("weight_decay", 1e-5)

        # Architecture params (used during fit when n_features is known)
        self.n_layers = kwargs.pop("n_layers", 2)
        self.d_hidden = kwargs.pop("d_hidden", 64)
        self.degree = kwargs.pop("degree", 4)

    def _build_model(self, n_features: int):
        torch.manual_seed(self.seed)
        d_out = 1 if self.task_type in ("binary", "regression") else self.n_classes

        self.model = _TabKANNet(
            n_features=n_features,
            d_hidden=self.d_hidden,
            n_layers=self.n_layers,
            degree=self.degree,
            d_out=d_out,
        )
        self.model.to(self.device)

    def _get_loss_fn(self):
        if self.task_type == "binary":
            return nn.BCEWithLogitsLoss()
        elif self.task_type == "multiclass":
            return nn.CrossEntropyLoss()
        else:
            return nn.MSELoss()

    def _make_loader(self, X, y=None, shuffle=False):
        tensors = [torch.tensor(X, dtype=torch.float32)]
        if y is not None:
            dtype = torch.long if self.task_type == "multiclass" else torch.float32
            tensors.append(torch.tensor(y, dtype=dtype))
        ds = TensorDataset(*tensors)
        generator = torch.Generator().manual_seed(self.seed) if shuffle else None
        return DataLoader(ds, batch_size=self.batch_size, shuffle=shuffle, generator=generator)

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        n_features = X_train.shape[1]
        self._build_model(n_features)

        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay,
        )
        loss_fn = self._get_loss_fn()
        train_loader = self._make_loader(X_train, y_train, shuffle=True)

        best_val_loss = float("inf")
        patience_counter = 0
        best_state = None

        for epoch in range(self.max_epochs):
            self.model.train()
            train_loss = 0.0
            for batch in train_loader:
                X_batch = batch[0].to(self.device)
                y_batch = batch[1].to(self.device)

                optimizer.zero_grad()
                out = self.model(X_batch).squeeze(-1)

                loss = loss_fn(out, y_batch)

                loss.backward()
                optimizer.step()
                train_loss += loss.item() * X_batch.size(0)

            train_loss /= len(train_loader.dataset)

            if X_val is not None and y_val is not None:
                val_loss = self._evaluate_loss(X_val, y_val, loss_fn)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                else:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        logger.debug(f"Early stopping at epoch {epoch+1}")
                        break

        if best_state is not None:
            self.model.load_state_dict(best_state)
            self.model.to(self.device)

        self.is_fitted = True
        return self

    def _evaluate_loss(self, X, y, loss_fn):
        self.model.eval()
        loader = self._make_loader(X, y, shuffle=False)
        total_loss = 0.0
        with torch.no_grad():
            for batch in loader:
                X_batch = batch[0].to(self.device)
                y_batch = batch[1].to(self.device)
                out = self.model(X_batch).squeeze(-1)
                loss = loss_fn(out, y_batch)
                total_loss += loss.item() * X_batch.size(0)
        return total_loss / len(loader.dataset)

    def predict(self, X):
        self.model.eval()
        loader = self._make_loader(X, shuffle=False)
        preds = []
        with torch.no_grad():
            for (X_batch,) in loader:
                X_batch = X_batch.to(self.device)
                out = self.model(X_batch).squeeze(-1)
                if self.task_type == "binary":
                    preds.append((torch.sigmoid(out) > 0.5).cpu().numpy().astype(int))
                elif self.task_type == "multiclass":
                    preds.append(out.argmax(dim=1).cpu().numpy())
                else:
                    preds.append(out.cpu().numpy())
        return np.concatenate(preds)

    def predict_proba(self, X):
        if self.task_type == "regression":
            raise NotImplementedError("predict_proba not available for regression")
        self.model.eval()
        loader = self._make_loader(X, shuffle=False)
        probs = []
        with torch.no_grad():
            for (X_batch,) in loader:
                X_batch = X_batch.to(self.device)
                out = self.model(X_batch).squeeze(-1)
                if self.task_type == "binary":
                    p = torch.sigmoid(out).cpu().numpy()
                    probs.append(np.column_stack([1 - p, p]))
                else:
                    probs.append(torch.softmax(out, dim=1).cpu().numpy())
        return np.concatenate(probs)
