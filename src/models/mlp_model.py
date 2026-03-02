"""Simple MLP baseline model with custom PyTorch training loop."""

import logging

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.base import BaseModel

logger = logging.getLogger("tabular_benchmark")


class _MLPBlock(nn.Module):
    """Single MLP block: Linear -> BatchNorm -> ReLU -> Dropout."""

    def __init__(self, d_in: int, d_out: int, dropout: float):
        super().__init__()
        self.linear = nn.Linear(d_in, d_out)
        self.bn = nn.BatchNorm1d(d_out)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.relu(self.bn(self.linear(x))))


class _MLP(nn.Module):
    """MLP: Input -> [Linear -> BatchNorm -> ReLU -> Dropout] x n_blocks -> Linear output."""

    def __init__(self, n_features: int, d_hidden: int, n_blocks: int, dropout: float, d_out: int):
        super().__init__()
        layers = []
        d_in = n_features
        for _ in range(n_blocks):
            layers.append(_MLPBlock(d_in, d_hidden, dropout))
            d_in = d_hidden
        self.blocks = nn.Sequential(*layers)
        self.head = nn.Linear(d_hidden, d_out)

    def forward(self, x):
        return self.head(self.blocks(x))


class MLPModel(BaseModel):
    MODEL_NAME = "mlp"
    FAMILY = "deep_learning"
    SUPPORTS_GPU = True

    def __init__(self, task_type: str, n_classes: int | None = None, seed: int = 42, **kwargs):
        super().__init__(task_type, n_classes, seed=seed, **kwargs)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_epochs = kwargs.pop("max_epochs", 200)
        self.patience = kwargs.pop("patience", 20)
        self.batch_size = kwargs.pop("batch_size", 256)
        self.lr = kwargs.pop("learning_rate", 1e-4)
        self.weight_decay = kwargs.pop("weight_decay", 1e-5)

        # Architecture params (will be used during fit when we know n_features)
        self.n_blocks = kwargs.pop("n_blocks", 2)
        self.d_hidden = kwargs.pop("d_hidden", 256)
        self.dropout = kwargs.pop("dropout", 0.1)

    def _build_model(self, n_features: int):
        torch.manual_seed(self.seed)
        d_out = 1 if self.task_type in ("binary", "regression") else self.n_classes

        self.model = _MLP(
            n_features=n_features,
            d_hidden=self.d_hidden,
            n_blocks=self.n_blocks,
            dropout=self.dropout,
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
            # Training
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

            # Validation
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
