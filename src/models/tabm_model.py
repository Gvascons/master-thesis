"""TabM model wrapper using the official ``tabm`` package.

Based on "TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling"
(Gorishniy et al., ICLR 2025).  The official package provides LinearBatchEnsemble
layers that share most weights across K ensemble members while using per-member
scaling adapters (R, S, B) — this acts as effective regularization and is a key
design choice of the paper.

The implementation uses ``tabm.TabM.make()`` which sets up:
  - EnsembleView  -> creates K copies of each input
  - MLPBackboneBatchEnsemble  -> shared-weight MLP blocks with per-member adapters
  - LinearEnsemble  -> independent output heads per member

Training optimises the **mean of per-member losses** (not the loss of the mean
prediction), following the paper's prescription.
"""

import logging

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.base import BaseModel

logger = logging.getLogger("tabular_benchmark")


class TabMModel(BaseModel):
    MODEL_NAME = "tabm"
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

        # TabM architecture params (passed to tabm.TabM.make)
        self.k = kwargs.pop("k", 32)
        self.d_block = kwargs.pop("d_block", 256)
        self.n_blocks = kwargs.pop("n_blocks", 3)
        self.dropout = kwargs.pop("dropout", 0.1)
        self.arch_type = kwargs.pop("arch_type", "tabm")

    def _build_model(self, n_features: int):
        """Build a TabM model using the official ``tabm`` package."""
        torch.manual_seed(self.seed)
        import tabm as tabm_pkg

        d_out = 1 if self.task_type in ("binary", "regression") else self.n_classes

        self.model = tabm_pkg.TabM.make(
            n_num_features=n_features,
            cat_cardinalities=None,
            d_out=d_out,
            k=self.k,
            d_block=self.d_block,
            n_blocks=self.n_blocks,
            dropout=self.dropout,
            arch_type=self.arch_type,
        )
        self.model.to(self.device)

    def _get_loss_fn(self):
        if self.task_type == "binary":
            return nn.BCEWithLogitsLoss()
        elif self.task_type == "multiclass":
            return nn.CrossEntropyLoss()
        return nn.MSELoss()

    def _compute_per_member_loss(self, preds, targets, loss_fn):
        """Compute loss per ensemble member and return the mean.

        As specified in the paper: "the mean loss must be optimised,
        not the loss of the mean prediction".

        Args:
            preds: (batch, k, d_out) — one prediction per member.
            targets: (batch,)
        """
        k = preds.size(1)
        total_loss = 0.0
        for i in range(k):
            member_pred = preds[:, i].squeeze(-1)  # (batch,) or (batch, n_classes)
            total_loss = total_loss + loss_fn(member_pred, targets)
        return total_loss / k

    def _make_loader(self, X, y=None, shuffle=False):
        tensors = [torch.tensor(X, dtype=torch.float32)]
        if y is not None:
            dtype = torch.long if self.task_type == "multiclass" else torch.float32
            tensors.append(torch.tensor(y, dtype=dtype))
        generator = torch.Generator().manual_seed(self.seed) if shuffle else None
        return DataLoader(
            TensorDataset(*tensors),
            batch_size=self.batch_size,
            shuffle=shuffle,
            generator=generator,
        )

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        self._build_model(X_train.shape[1])
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
            for batch in train_loader:
                X_b = batch[0].to(self.device)
                y_b = batch[1].to(self.device)
                optimizer.zero_grad()
                # Forward: returns (batch, k, d_out)
                preds = self.model(x_num=X_b)
                loss = self._compute_per_member_loss(preds, y_b, loss_fn)
                loss.backward()
                optimizer.step()

            # Early stopping using per-member loss on validation set
            if X_val is not None and y_val is not None:
                val_loss = self._eval_per_member_loss(X_val, y_val, loss_fn)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                else:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        logger.debug(f"TabM early stopping at epoch {epoch + 1}")
                        break

        if best_state is not None:
            self.model.load_state_dict(best_state)
            self.model.to(self.device)

        self.is_fitted = True
        return self

    def _eval_per_member_loss(self, X, y, loss_fn):
        """Evaluate per-member loss (not loss-of-mean) on a dataset."""
        self.model.eval()
        loader = self._make_loader(X, y)
        total = 0.0
        n = 0
        with torch.no_grad():
            for batch in loader:
                X_b = batch[0].to(self.device)
                y_b = batch[1].to(self.device)
                preds = self.model(x_num=X_b)  # (batch, k, d_out)
                total += self._compute_per_member_loss(preds, y_b, loss_fn).item() * X_b.size(0)
                n += X_b.size(0)
        return total / n

    def _predict_mean(self, X):
        """Average predictions across the K ensemble members."""
        self.model.eval()
        loader = self._make_loader(X)
        results = []
        with torch.no_grad():
            for (X_b,) in loader:
                preds = self.model(x_num=X_b.to(self.device))  # (batch, k, d_out)
                mean_pred = preds.mean(dim=1).squeeze(-1)  # (batch,) or (batch, n_classes)
                results.append(mean_pred)
        return torch.cat(results, dim=0)

    def predict(self, X):
        mean_pred = self._predict_mean(X)
        if self.task_type == "binary":
            return (torch.sigmoid(mean_pred) > 0.5).cpu().numpy().astype(int)
        elif self.task_type == "multiclass":
            return mean_pred.argmax(dim=1).cpu().numpy()
        else:
            return mean_pred.cpu().numpy()

    def predict_proba(self, X):
        if self.task_type == "regression":
            raise NotImplementedError("predict_proba not available for regression")
        mean_pred = self._predict_mean(X)
        if self.task_type == "binary":
            p = torch.sigmoid(mean_pred).cpu().numpy()
            return np.column_stack([1 - p, p])
        else:
            return torch.softmax(mean_pred, dim=1).cpu().numpy()
