# SAINT (Self-Attention and Intersample Attention Transformer)

> **Paper**: Somepalli et al., "SAINT: Improved Neural Networks for Tabular Data via Row Attention and Contrastive Pre-Training", 2021  
> **Role in this thesis**: Extends FT-Transformer with a second attention mechanism — **inter-sample (row) attention** — that allows samples to attend to each other within a batch, not just features.

---

## 1. What It Is

SAINT builds on the Transformer-for-tabular approach (like FT-Transformer) but adds a crucial second dimension of attention:

1. **Self-Attention (column attention)**: Standard feature-to-feature attention (same as FT-Transformer)
2. **Inter-Sample Attention (row attention)**: Samples attend to other samples within the batch

This dual attention means SAINT can learn both **feature interactions** (which features relate?) and **sample interactions** (which data points are similar?). The inter-sample attention is conceptually similar to k-nearest-neighbors but learned end-to-end.

---

## 2. Architecture

### 2.1 Dual Attention Block

Each SAINT block applies two attention stages sequentially:

```
Input tokens (batch × features × d)
  → LayerNorm → Self-Attention (across features) + residual
  → LayerNorm → GEGLU Feed-Forward + residual
  → LayerNorm → Inter-Sample Attention (across batch) + residual
  → LayerNorm → GEGLU Feed-Forward + residual
  → Output tokens
```

### 2.2 Self-Attention (Column-wise)

Same as FT-Transformer: each feature token attends to all other feature tokens within the same sample. Captures pairwise feature interactions.

### 2.3 Inter-Sample Attention (Row-wise)

The key innovation. For each feature position, attention is computed **across samples in the batch**:

```
For feature j:
  Q = sample_i's token_j    → "What do I need?"
  K = all samples' token_j   → "What do others offer?"
  V = all samples' token_j   → "What information do others carry?"
  
  output_i = softmax(Q·Kᵀ/√d) · V
  → sample i's representation is informed by similar samples
```

**Intuition**: If sample A (age=45, income=80k) is in the batch with sample B (age=44, income=82k), their row attention weights will be high — B's label information implicitly helps A's prediction, similar to how kNN works.

### 2.4 GEGLU Activation

SAINT uses GEGLU (GELU-Gated Linear Unit, Shazeer 2020) instead of standard ReLU or GELU in its feed-forward blocks:

```
GEGLU(x) = x₁ ⊙ GELU(x₂)    (split input in half, gate one half with GELU of the other)
```

The gating mechanism allows the network to learn which dimensions to activate, providing richer expressivity than fixed activation functions.

---

## 3. Contrastive Pre-Training (CutMix + Mixup)

The paper also proposes a contrastive pre-training scheme:
- **CutMix**: Replace random features of a sample with features from another sample
- **Mixup**: Interpolate between samples
- **Contrastive loss**: Learn representations that distinguish original from augmented samples

**Note**: This pre-training is not used in this thesis — all models are trained from scratch for fair comparison.

---

## 4. Hyperparameters (Search Space in This Thesis)

| Parameter | Range | Scale | Effect |
|-----------|-------|-------|--------|
| `depth` | 1–6 | Linear | Number of dual-attention blocks |
| `heads` | {4, 8} | Categorical | Attention heads |
| `dim` | {32, 64, 128, 256} | Categorical | Token embedding dimension |
| `attn_dropout` | 0.0–0.4 | Linear | Dropout on attention weights |
| `ff_dropout` | 0.0–0.4 | Linear | Dropout in feed-forward layers |
| `learning_rate` | 1e-5–1e-3 | Log | AdamW learning rate |

---

## 5. Strengths and Weaknesses

### Strengths
- **Inter-sample attention is unique**: No other model in this benchmark captures sample-to-sample relationships
- **Richer context**: Each prediction is informed by similar examples in the batch (implicit nearest-neighbor effect)
- **Feature + sample interactions**: Two dimensions of attention capture different types of patterns

### Weaknesses
- **Batch-dependent predictions**: Changing the batch composition changes predictions — introduces stochasticity at inference
- **Quadratic in batch size**: Inter-sample attention is O(B²) — expensive for large batches
- **Training instability**: Two attention mechanisms double the optimization difficulty
- **Empirical results mixed**: Doesn't consistently outperform simpler approaches (Gorishniy et al., 2021)

---

## 6. Key References

| Reference | Contribution |
|-----------|--------------|
| Somepalli et al. (2021) | **SAINT** — dual attention with inter-sample mechanism |
| Gorishniy et al. (2021) | FT-Transformer — the single-attention baseline SAINT extends |
| Shazeer (2020) | GLU variants (GEGLU) used in SAINT's feed-forward blocks |

---

## 7. In This Thesis

SAINT tests whether **inter-sample attention** provides value beyond feature-only attention (FT-Transformer). The comparison is direct: same Transformer backbone, but SAINT adds row attention.

**Preprocessing**: Standardize numericals, one-hot encode categoricals, median-impute.
