"""FT-Transformer model wrapper with custom PyTorch training loop."""

import logging

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.base import BaseModel

logger = logging.getLogger("tabular_benchmark")


class FTTransformerModel(BaseModel):
    MODEL_NAME = "ft_transformer"
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

        # Architecture params (will be set during fit when we know n_features)
        self.n_blocks = kwargs.pop("n_blocks", 3)
        self.d_block = kwargs.pop("d_block", 192)
        # Accept legacy "d_token" as an alias for "d_block"
        if "d_token" in kwargs:
            self.d_block = kwargs.pop("d_token")
        self.attention_n_heads = kwargs.pop("attention_n_heads", 8)
        self.attention_dropout = kwargs.pop("attention_dropout", 0.2)
        self.ffn_d_hidden_multiplier = kwargs.pop("ffn_d_hidden_multiplier", 2.0)
        self.ffn_dropout = kwargs.pop("ffn_dropout", 0.1)
        self.residual_dropout = kwargs.pop("residual_dropout", 0.0)

    def _build_model(self, n_features: int):
        """Build the FT-Transformer using rtdl_revisiting_models.

        NOTE: Currently all features arrive as continuous (categoricals are
        one-hot encoded during DL preprocessing). To leverage FT-Transformer's
        native categorical embeddings on mixed datasets, the following changes
        would be needed:
          1. Add a preprocessing path that ordinal-encodes categoricals and
             returns separate (X_num, X_cat) arrays plus cat_cardinalities.
          2. Extend PreprocessedData to carry X_num, X_cat, and cardinalities.
          3. Pass X_cat tensor to self.model(x_cont, x_cat) in fit/predict.
        This is a significant performance opportunity on mixed-feature datasets
        like adult, bank_marketing, diamonds, and credit_g.
        """
        import rtdl_revisiting_models

        d_out = 1 if self.task_type in ("binary", "regression") else self.n_classes

        self.model = rtdl_revisiting_models.FTTransformer(
            n_cont_features=n_features,
            cat_cardinalities=[],
            d_out=d_out,
            n_blocks=self.n_blocks,
            d_block=self.d_block,
            attention_n_heads=self.attention_n_heads,
            attention_dropout=self.attention_dropout,
            ffn_d_hidden_multiplier=self.ffn_d_hidden_multiplier,
            ffn_dropout=self.ffn_dropout,
            residual_dropout=self.residual_dropout,
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
                out = self.model(X_batch, None).squeeze(-1)

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
                out = self.model(X_batch, None).squeeze(-1)
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
                out = self.model(X_batch, None).squeeze(-1)
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
                out = self.model(X_batch, None).squeeze(-1)
                if self.task_type == "binary":
                    p = torch.sigmoid(out).cpu().numpy()
                    probs.append(np.column_stack([1 - p, p]))
                else:
                    probs.append(torch.softmax(out, dim=1).cpu().numpy())
        return np.concatenate(probs)
