"""Self-contained STab (Stochastic Transformers for Tabular Data) implementation.

Based on the paper: "Transformers with Stochastic Competition for Tabular Data
Modelling" (Voskou, Christoforou & Chatzis, ICML 2024 Workshop).

Reference implementation:
  https://github.com/avoskou/Transformers-with-Stochastic-Competition-for-Tabular-Data-Modelling
"""

import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from torch.utils.data import DataLoader, TensorDataset

from src.models.base import BaseModel

logger = logging.getLogger("tabular_benchmark")


# ---------- LWTA (Local Winner Takes All) ----------


def _concrete_sample(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Gumbel-Softmax sample (concrete relaxation)."""
    U = torch.rand_like(logits)
    G = -torch.log(-torch.log(U + 1e-8) + 1e-8)
    t = (logits + G) / temperature
    return F.softmax(t, dim=-1)


class LWTA(nn.Module):
    """Local Winner Takes All activation.

    Partitions the input into blocks of size U. Within each block, only the
    "winning" neuron is retained via Gumbel-Softmax sampling. All other neurons
    are zeroed out, promoting sparse, stochastic representations.
    """

    def __init__(self, U: int = 2, kl_weight: float = 1.0, temperature: float = 0.69,
                 temp_test: float = 0.01, return_mask: bool = False):
        super().__init__()
        self.U = U
        self.kl_weight = kl_weight
        self.temperature = temperature
        self.temp_test = temp_test
        self.return_mask = return_mask
        self.kl_loss = 0.0

    def forward(self, x: torch.Tensor):
        original_shape = x.shape
        # Reshape into blocks of size U
        logits = x.reshape(-1, x.size(-1) // self.U, self.U)

        if self.training:
            mask = _concrete_sample(logits, self.temperature)
        else:
            mask = _concrete_sample(logits, self.temp_test)

        mask_r = mask.reshape(original_shape)

        # Compute KL divergence for regularization during training
        if self.training:
            q = mask
            log_q = torch.log(q + 1e-8)
            log_p = torch.log(torch.tensor(1.0 / self.U, device=x.device))
            kl = torch.sum(q * (log_q - log_p), dim=-1)
            self.kl_loss = torch.mean(kl) / 1000.0
        else:
            self.kl_loss = 0.0

        out = x * mask_r

        if self.return_mask:
            return out, mask_r
        return out


# ---------- MC-Dropout (active at inference too) ----------


class MCDropout(nn.Module):
    """Dropout that stays active during inference for Bayesian averaging."""

    def __init__(self, p: float = 0.1):
        super().__init__()
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Always apply dropout (MC-dropout for Bayesian inference)
        return F.dropout(x, self.p, training=True)


# ---------- LocalLinear (per-feature linear projection) ----------


class LocalLinear(nn.Module):
    """Per-feature linear projection (TimeDistributed-style)."""

    def __init__(self, in_size: int, out_size: int, n_features: int):
        super().__init__()
        self.w = nn.Parameter(torch.randn(n_features, in_size, out_size))
        self.b = nn.Parameter(torch.randn(1, n_features, out_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, n_features, in_size)
        x = x.transpose(0, 1)  # (n_features, batch, in_size)
        x = torch.bmm(x, self.w)  # (n_features, batch, out_size)
        x = x.transpose(0, 1)  # (batch, n_features, out_size)
        return x + self.b


# ---------- Embedding Mixture Layer ----------


class EmbeddingMixture(nn.Module):
    """Probabilistic Embedding Mixture for numerical features.

    Maintains J alternative linear embeddings per feature and selects among
    them via Gumbel-Softmax competition based on the input value.
    """

    def __init__(self, dim: int, n_numerical: int, J: int = 16):
        super().__init__()
        self.J = J
        self.n_numerical = n_numerical
        self.dim = dim

        # J alternative embedding weight+bias pairs per feature
        self.weights = nn.Parameter(torch.randn(n_numerical, J))
        self.biases = nn.Parameter(torch.randn(n_numerical, J))
        self.w = nn.Parameter(torch.randn(n_numerical, J, dim))
        self.b = nn.Parameter(torch.randn(n_numerical, J, dim))

        # LWTA-style selection among alternatives
        self.lwta = LWTA(U=J, kl_weight=1.0, return_mask=True) if J > 1 else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, n_numerical) numerical feature values

        Returns:
            (batch, n_numerical, dim) embedded features
        """
        x = rearrange(x, 'b n -> b n 1')
        x = x * self.weights + self.biases  # (batch, n_numerical, J)

        if self.J > 1 and self.lwta is not None:
            x, mask = self.lwta(x)
            # mask: (batch, n_numerical, J)
            mask = mask.transpose(0, 1)  # (n_numerical, batch, J)
            x = x.transpose(0, 1)  # (n_numerical, batch, J)
            b = torch.bmm(mask, self.b).transpose(0, 1)  # (batch, n_numerical, dim)
            x = torch.bmm(x, self.w)  # (n_numerical, batch, dim)
            x = x.transpose(0, 1)  # (batch, n_numerical, dim)
            y = x + b
        else:
            x = x.transpose(0, 1)
            x = torch.bmm(x, self.w)
            x = x.transpose(0, 1)
            b = self.b.squeeze(-2) if self.J == 1 else self.b[:, 0, :]
            y = x + b

        return y

    @property
    def kl_loss(self) -> float:
        if self.lwta is not None:
            return self.lwta.kl_loss
        return 0.0


# ---------- GlobalResnet (Parallel Aggregation Module) ----------


class GlobalResnet(nn.Module):
    """Parallel fully-connected aggregation module.

    Projects d-dimensional token embeddings back to scalars, aggregates them,
    and processes via LWTA-based layers. Output is added to the CLS token.
    """

    def __init__(self, dim: int, n_features: int, dropout: float = 0.2, U: int = 2):
        super().__init__()
        self.ln1 = LocalLinear(dim, 1, n_features)
        self.ln2 = nn.Linear(n_features, 4 * dim)
        self.ln3 = nn.Linear(4 * dim, dim)
        self.lwta = LWTA(U)
        self.norm = nn.LayerNorm(n_features)
        self.norm2 = nn.LayerNorm(dim)
        self.drop1 = MCDropout(dropout)
        self.drop2 = MCDropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ln1(x)
        x = rearrange(x, 'b n 1 -> b n')
        x = self.norm(x)
        x = self.drop1(x)
        x = self.ln2(x)
        x = self.lwta(x)
        x = self.ln3(x)
        x = rearrange(x, 'b n -> b 1 n')
        x = self.norm2(x)
        return self.drop2(x)

    @property
    def kl_loss(self):
        return self.lwta.kl_loss


# ---------- Feed-Forward with LWTA ----------


class LWTAFeedForward(nn.Module):
    """Feed-forward block using LWTA activation instead of ReLU/GELU."""

    def __init__(self, dim: int, mult: int = 4, dropout: float = 0.0, U: int = 2):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.linear1 = nn.Linear(dim, dim * mult)
        self.lwta = LWTA(U)
        self.drop = MCDropout(dropout)
        self.linear2 = nn.Linear(dim * mult, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        x = self.linear1(x)
        x = self.lwta(x)
        x = self.drop(x)
        return self.linear2(x)

    @property
    def kl_loss(self):
        return self.lwta.kl_loss


# ---------- Attention with Learned Bias ----------


class BiasedAttention(nn.Module):
    """Multi-head self-attention with learnable pairwise feature bias."""

    def __init__(self, dim: int, heads: int = 8, dim_head: int = 32,
                 dropout: float = 0.0, n_features: int = 10):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)
        self.dropout = MCDropout(dropout)
        # Learned attention bias exploiting fixed tabular feature ordering
        self.attention_bias = nn.Parameter(torch.zeros(1, heads, n_features, n_features))

    def forward(self, x: torch.Tensor):
        h = self.heads
        x = self.norm(x)
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = (rearrange(t, 'b n (h d) -> b h n d', h=h) for t in (q, k, v))

        q = q * self.scale
        sim = torch.einsum('b h i d, b h j d -> b h i j', q, k)
        sim = sim + self.attention_bias

        attn = sim.softmax(dim=-1)
        attn = self.dropout(attn)

        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)', h=h)
        return self.to_out(out)


# ---------- Hybrid Transformer Layer ----------


class HybridTransformerLayer(nn.Module):
    """A single Hybrid Transformer layer.

    Combines standard self-attention + LWTA feed-forward with a parallel
    GlobalResnet aggregation module that feeds into the CLS token.
    """

    def __init__(self, dim: int, heads: int, dim_head: int,
                 attn_dropout: float, ff_dropout: float,
                 n_features: int, U: int = 2):
        super().__init__()
        self.attn = BiasedAttention(dim, heads, dim_head, attn_dropout, n_features)
        self.ff = LWTAFeedForward(dim, dropout=ff_dropout, U=U)
        self.globres = GlobalResnet(dim, n_features, dropout=ff_dropout, U=U)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Parallel aggregation module (runs on pre-attention tokens)
        dx = self.globres(x)

        # Main Transformer path
        x = self.attn(x) + x
        x = self.ff(x) + x

        # Add parallel module output to CLS token only
        x[:, :1] = dx + x[:, :1]
        return x

    @property
    def kl_loss(self):
        return self.ff.kl_loss + self.globres.kl_loss


# ---------- Full STab Network ----------


class STabNet(nn.Module):
    """STab: Stochastic Transformer for Tabular Data.

    Full architecture combining Embedding Mixture, Hybrid Transformer layers,
    and classification/regression head.
    """

    def __init__(
        self,
        n_features: int,
        d_out: int,
        dim: int = 128,
        depth: int = 4,
        heads: int = 8,
        dim_head: int = 32,
        attn_dropout: float = 0.1,
        ff_dropout: float = 0.1,
        U: int = 2,
        cases: int = 16,
    ):
        super().__init__()
        assert dim % heads == 0, f"dim ({dim}) must be divisible by heads ({heads})"
        self.n_features = n_features
        self.dim = dim
        self.depth = depth

        # Numerical Embedding Mixture (all features treated as numerical after preprocessing)
        self.numerical_embedder = EmbeddingMixture(dim, n_features, J=cases)

        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

        # n_features + 1 (CLS) for attention bias
        total_tokens = n_features + 1

        # Hybrid Transformer layers
        self.layers = nn.ModuleList([
            HybridTransformerLayer(
                dim=dim, heads=heads, dim_head=dim_head,
                attn_dropout=attn_dropout, ff_dropout=ff_dropout,
                n_features=total_tokens, U=U,
            )
            for _ in range(depth)
        ])

        # Output head
        self.head = nn.Sequential(
            MCDropout(ff_dropout),
            nn.LayerNorm(dim),
            nn.Linear(dim, d_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, n_features) numerical input values

        Returns:
            (batch, d_out) logits
        """
        b = x.size(0)

        # Embed numerical features via Mixture Layer
        x = self.numerical_embedder(x)  # (batch, n_features, dim)

        # Prepend CLS token
        cls = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        x = torch.cat([cls, x], dim=1)  # (batch, 1 + n_features, dim)

        # Hybrid Transformer layers
        for layer in self.layers:
            x = layer(x)

        # Use CLS token for prediction
        cls_out = x[:, 0]
        return self.head(cls_out)

    def kl_loss(self) -> torch.Tensor:
        """Total KL divergence from all stochastic components."""
        kl = 0.0
        # Embedding mixture KL
        emb_kl = self.numerical_embedder.kl_loss
        if isinstance(emb_kl, torch.Tensor):
            kl = kl + emb_kl
        else:
            kl = kl + emb_kl

        # Layer KL (LWTA in FF and GlobalResnet)
        for layer in self.layers:
            layer_kl = layer.kl_loss
            if isinstance(layer_kl, torch.Tensor):
                kl = kl + layer_kl
            else:
                kl = kl + layer_kl

        return kl


# ---------- Model Wrapper ----------


class STabModel(BaseModel):
    """STab model wrapper implementing the BaseModel interface.

    Trains a Stochastic Transformer with LWTA activations, Embedding Mixture,
    and Hybrid Transformer layers. Uses Bayesian averaging at inference time.
    """

    MODEL_NAME = "stab"
    FAMILY = "deep_learning"
    SUPPORTS_GPU = True

    def __init__(self, task_type: str, n_classes: int | None = None,
                 seed: int = 42, **kwargs):
        super().__init__(task_type, n_classes, seed=seed, **kwargs)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_epochs = kwargs.pop("max_epochs", 200)
        self.patience = kwargs.pop("patience", 20)
        self.batch_size = kwargs.pop("batch_size", 256)
        self.lr = kwargs.pop("learning_rate", 1e-3)
        self.weight_decay = kwargs.pop("weight_decay", 1e-4)

        # Architecture hyperparameters
        self.depth = kwargs.pop("depth", 4)
        self.heads = kwargs.pop("heads", 8)
        self.dim = kwargs.pop("dim", 128)
        self.attn_dropout = kwargs.pop("attn_dropout", 0.25)
        self.ff_dropout = kwargs.pop("ff_dropout", 0.25)
        self.cases = kwargs.pop("cases", 16)
        self.lwta_U = kwargs.pop("lwta_block_size", 2)

        # Bayesian inference
        self.n_inference_samples = kwargs.pop("n_inference_samples", 64)

        # KL loss weight
        self.kl_weight = kwargs.pop("kl_weight", 0.01)

        # Warmup epochs
        self.warmup_epochs = kwargs.pop("warmup_epochs", 10)

    def _build_model(self, n_features: int):
        torch.manual_seed(self.seed)
        d_out = 1 if self.task_type in ("binary", "regression") else self.n_classes

        # Ensure dim is divisible by heads
        dim = self.dim
        if dim % self.heads != 0:
            dim = (dim // self.heads) * self.heads
            if dim == 0:
                dim = self.heads
            logger.warning(f"Adjusted dim from {self.dim} to {dim} for head divisibility")

        # Ensure dim * 4 (FF expansion) is divisible by U (LWTA block size)
        ff_dim = dim * 4
        if ff_dim % self.lwta_U != 0:
            logger.warning(f"FF dim {ff_dim} not divisible by U={self.lwta_U}, adjusting")

        self.model = STabNet(
            n_features=n_features,
            d_out=d_out,
            dim=dim,
            depth=self.depth,
            heads=self.heads,
            dim_head=dim // self.heads,
            attn_dropout=self.attn_dropout,
            ff_dropout=self.ff_dropout,
            U=self.lwta_U,
            cases=self.cases,
        )
        self.model.to(self.device)

    def _get_task_loss_fn(self):
        if self.task_type == "binary":
            return nn.BCEWithLogitsLoss()
        elif self.task_type == "multiclass":
            return nn.CrossEntropyLoss()
        return nn.MSELoss()

    def _compute_loss(self, out, y_batch, task_loss_fn):
        """Combined task loss + KL divergence loss."""
        task_loss = task_loss_fn(out, y_batch)
        kl_loss = self.model.kl_loss()
        if isinstance(kl_loss, torch.Tensor):
            combined = (1 - self.kl_weight) * task_loss + self.kl_weight * kl_loss
        else:
            combined = task_loss
        return combined

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
        task_loss_fn = self._get_task_loss_fn()
        train_loader = self._make_loader(X_train, y_train, shuffle=True)

        # Phase 1: Warmup with linear LR schedule
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay,
        )
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.001, total_iters=self.warmup_epochs,
        )

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
                loss = self._compute_loss(out, y_batch, task_loss_fn)
                loss.backward()
                optimizer.step()

            # LR scheduling
            if epoch < self.warmup_epochs:
                warmup_scheduler.step()

            # Validation-based early stopping and LR plateau reduction
            if X_val is not None and y_val is not None:
                val_loss = self._eval_loss(X_val, y_val, task_loss_fn)

                if epoch >= self.warmup_epochs:
                    # ReduceLROnPlateau after warmup
                    if not hasattr(self, '_plateau_scheduler'):
                        self._plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                            optimizer, mode='min', patience=5, factor=0.5, min_lr=1e-5,
                        )
                    self._plateau_scheduler.step(val_loss)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_state = {
                        k: v.cpu().clone() for k, v in self.model.state_dict().items()
                    }
                else:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        logger.debug(f"STab early stopping at epoch {epoch + 1}")
                        break

        # Clean up scheduler reference
        if hasattr(self, '_plateau_scheduler'):
            del self._plateau_scheduler

        if best_state is not None:
            self.model.load_state_dict(best_state)
            self.model.to(self.device)

        self.is_fitted = True
        return self

    def _eval_loss(self, X, y, task_loss_fn):
        """Evaluate loss without gradient computation."""
        self.model.eval()
        loader = self._make_loader(X, y)
        total = 0.0
        n = 0
        with torch.no_grad():
            for batch in loader:
                X_b = batch[0].to(self.device)
                y_b = batch[1].to(self.device)
                out = self.model(X_b).squeeze(-1)
                total += task_loss_fn(out, y_b).item() * X_b.size(0)
                n += X_b.size(0)
        return total / n

    def _bayesian_forward(self, X: np.ndarray) -> torch.Tensor:
        """Run N stochastic forward passes and average the outputs (Bayesian averaging)."""
        self.model.eval()  # Keep eval mode, but MC-Dropout and LWTA remain stochastic
        loader = self._make_loader(X)
        all_outputs = []

        with torch.no_grad():
            for (X_batch,) in loader:
                X_b = X_batch.to(self.device)
                # Replicate for N samples and average
                batch_outputs = []
                for _ in range(self.n_inference_samples):
                    out = self.model(X_b).squeeze(-1)
                    batch_outputs.append(out)
                # Average over N samples
                avg_out = torch.stack(batch_outputs, dim=0).mean(dim=0)
                all_outputs.append(avg_out.cpu())

        return torch.cat(all_outputs, dim=0)

    def predict(self, X: np.ndarray) -> np.ndarray:
        outputs = self._bayesian_forward(X)
        if self.task_type == "binary":
            return (torch.sigmoid(outputs) > 0.5).numpy().astype(int)
        elif self.task_type == "multiclass":
            return outputs.argmax(dim=1).numpy()
        else:
            return outputs.numpy()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.task_type == "regression":
            raise NotImplementedError("predict_proba not available for regression")
        outputs = self._bayesian_forward(X)
        if self.task_type == "binary":
            p = torch.sigmoid(outputs).numpy()
            return np.column_stack([1 - p, p])
        else:
            return torch.softmax(outputs, dim=1).numpy()
