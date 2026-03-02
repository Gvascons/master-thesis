"""Self-contained SAINT (Self-Attention and Intersample Attention Transformer) implementation.

Based on the paper: "SAINT: Improved Neural Networks for Tabular Data via Row Attention
and Contrastive Pre-Training" (Somepalli et al., 2021).
"""

import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from src.models.base import BaseModel

logger = logging.getLogger("tabular_benchmark")


# ---------- SAINT Architecture ----------

class GEGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim),
        )

    def forward(self, x):
        return self.net(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, dim, heads=8, dropout=0.0):
        super().__init__()
        self.heads = heads
        self.dim_head = dim // heads
        self.scale = self.dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Sequential(nn.Linear(dim, dim), nn.Dropout(dropout))

    def forward(self, x):
        b, n, _ = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = (t.view(b, n, self.heads, self.dim_head).transpose(1, 2) for t in qkv)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = attn @ v
        out = out.transpose(1, 2).reshape(b, n, -1)
        return self.to_out(out)


class SAINTLayer(nn.Module):
    """One SAINT block: self-attention (with CLS) + intersample attention (without CLS) + feed-forward."""

    def __init__(self, dim, heads, attn_dropout, ff_dropout):
        super().__init__()
        # Self-attention (column-wise)
        self.norm1 = nn.LayerNorm(dim)
        self.self_attn = MultiHeadAttention(dim, heads, attn_dropout)

        # Intersample attention (row-wise)
        self.norm2 = nn.LayerNorm(dim)
        self.inter_attn = MultiHeadAttention(dim, heads, attn_dropout)

        # Feed-forward
        self.norm3 = nn.LayerNorm(dim)
        self.ff = FeedForward(dim, dropout=ff_dropout)

    def forward(self, x, cls_token):
        # x: (batch, n_features, dim)
        # cls_token: (batch, 1, dim)

        # Self-attention over features (with CLS token prepended)
        x_with_cls = torch.cat([cls_token, x], dim=1)  # (batch, 1+n_features, dim)
        x_with_cls = self.self_attn(self.norm1(x_with_cls)) + x_with_cls
        cls_token = x_with_cls[:, :1]  # updated CLS
        x = x_with_cls[:, 1:]          # updated features

        # Intersample attention: transpose so samples attend to each other (no CLS)
        b, n, d = x.shape
        x_t = x.transpose(0, 1)  # (n_features, batch, dim)
        x_t = self.inter_attn(self.norm2(x_t)) + x_t
        x = x_t.transpose(0, 1)  # (batch, n_features, dim)

        # Feed-forward
        x = self.ff(self.norm3(x)) + x
        return x, cls_token


class SAINTNet(nn.Module):
    def __init__(self, n_features, d_out, dim=128, depth=3, heads=8,
                 attn_dropout=0.1, ff_dropout=0.1):
        super().__init__()
        assert dim % heads == 0, f"dim ({dim}) must be divisible by heads ({heads})"

        # Per-feature embeddings
        self.feature_embeds = nn.ModuleList([nn.Linear(1, dim) for _ in range(n_features)])
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

        self.layers = nn.ModuleList([
            SAINTLayer(dim, heads, attn_dropout, ff_dropout)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(dim)
        self.head = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.ReLU(),
            nn.Linear(dim // 2, d_out),
        )

    def forward(self, x):
        # x: (batch, n_features)
        b = x.size(0)
        # Embed each feature independently with per-feature linear layers
        embedded = [self.feature_embeds[i](x[:, i : i + 1]) for i in range(x.size(1))]
        x = torch.stack(embedded, dim=1)  # (batch, n_features, dim)

        cls = self.cls_token.expand(b, -1, -1)

        for layer in self.layers:
            x, cls = layer(x, cls)

        # Use CLS token for prediction
        out = self.norm(cls.squeeze(1))
        return self.head(out)


# ---------- Model Wrapper ----------

class SAINTModel(BaseModel):
    MODEL_NAME = "saint"
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

        self.depth = kwargs.pop("depth", 3)
        self.heads = kwargs.pop("heads", 8)
        self.dim = kwargs.pop("dim", 128)
        self.attn_dropout = kwargs.pop("attn_dropout", 0.1)
        self.ff_dropout = kwargs.pop("ff_dropout", 0.1)

    def _build_model(self, n_features):
        torch.manual_seed(self.seed)
        d_out = 1 if self.task_type in ("binary", "regression") else self.n_classes
        self.model = SAINTNet(
            n_features=n_features,
            d_out=d_out,
            dim=self.dim,
            depth=self.depth,
            heads=self.heads,
            attn_dropout=self.attn_dropout,
            ff_dropout=self.ff_dropout,
        )
        self.model.to(self.device)

    def _get_loss_fn(self):
        if self.task_type == "binary":
            return nn.BCEWithLogitsLoss()
        elif self.task_type == "multiclass":
            return nn.CrossEntropyLoss()
        return nn.MSELoss()

    def _make_loader(self, X, y=None, shuffle=False):
        tensors = [torch.tensor(X, dtype=torch.float32)]
        if y is not None:
            dtype = torch.long if self.task_type == "multiclass" else torch.float32
            tensors.append(torch.tensor(y, dtype=dtype))
        generator = torch.Generator().manual_seed(self.seed) if shuffle else None
        return DataLoader(TensorDataset(*tensors), batch_size=self.batch_size, shuffle=shuffle, generator=generator)

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
                X_batch = batch[0].to(self.device)
                y_batch = batch[1].to(self.device)
                optimizer.zero_grad()
                out = self.model(X_batch).squeeze(-1)
                loss = loss_fn(out, y_batch)
                loss.backward()
                optimizer.step()

            if X_val is not None and y_val is not None:
                val_loss = self._eval_loss(X_val, y_val, loss_fn)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                else:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        logger.debug(f"SAINT early stopping at epoch {epoch+1}")
                        break

        if best_state is not None:
            self.model.load_state_dict(best_state)
            self.model.to(self.device)

        self.is_fitted = True
        return self

    def _eval_loss(self, X, y, loss_fn):
        self.model.eval()
        loader = self._make_loader(X, y)
        total = 0.0
        with torch.no_grad():
            for batch in loader:
                X_b = batch[0].to(self.device)
                y_b = batch[1].to(self.device)
                out = self.model(X_b).squeeze(-1)
                total += loss_fn(out, y_b).item() * X_b.size(0)
        return total / len(loader.dataset)

    def predict(self, X):
        self.model.eval()
        loader = self._make_loader(X)
        preds = []
        with torch.no_grad():
            for (X_batch,) in loader:
                out = self.model(X_batch.to(self.device)).squeeze(-1)
                if self.task_type == "binary":
                    preds.append((torch.sigmoid(out) > 0.5).cpu().numpy().astype(int))
                elif self.task_type == "multiclass":
                    preds.append(out.argmax(dim=1).cpu().numpy())
                else:
                    preds.append(out.cpu().numpy())
        return np.concatenate(preds)

    def predict_proba(self, X):
        if self.task_type == "regression":
            raise NotImplementedError
        self.model.eval()
        loader = self._make_loader(X)
        probs = []
        with torch.no_grad():
            for (X_batch,) in loader:
                out = self.model(X_batch.to(self.device)).squeeze(-1)
                if self.task_type == "binary":
                    p = torch.sigmoid(out).cpu().numpy()
                    probs.append(np.column_stack([1 - p, p]))
                else:
                    probs.append(torch.softmax(out, dim=1).cpu().numpy())
        return np.concatenate(probs)
