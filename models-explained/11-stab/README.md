# STab (Stochastic Transformers for Tabular Data)

> **Paper**: Voskou, Christoforou & Chatzis, "Transformers with Stochastic Competition for Tabular Data Modelling", ICML 2024 Workshop on Structured Probabilistic Inference & Generative Modeling  
> **arXiv**: [2407.13238](https://arxiv.org/abs/2407.13238)  
> **Workshop paper**: [Table Representation Learning Workshop](https://table-representation-learning.github.io/assets/papers/stab_self_supervised_learning_.pdf)  
> **Code**: [github.com/avoskou/Transformers-with-Stochastic-Competition-for-Tabular-Data-Modelling](https://github.com/avoskou/Transformers-with-Stochastic-Competition-for-Tabular-Data-Modelling)  
> **Role in this thesis**: A Transformer-based architecture that introduces **stochastic competition** mechanisms — Local Winner Takes All (LWTA) activations and a novel Embedding Mixture Layer — combined with a **Hybrid Transformer** module that fuses self-attention with a parallel fully-connected aggregation path. STab tests whether stochasticity-driven regularization and architectural hybridization can close the gap between Transformers and GBDTs on tabular data.

---

## 1. What It Is

STab is a modified Transformer encoder designed specifically for tabular data. It builds on the FT-Transformer foundation but introduces three key innovations:

1. **Local Winner Takes All (LWTA)**: Replaces deterministic activations (ReLU/GELU) with stochastic competition — neurons within blocks compete, and only the "winner" is retained. This promotes sparsity and regularization without extra parameters.
2. **Embedding Mixture Layer**: Instead of a single linear projection per numerical feature, maintains J alternative linear embeddings and selects among them probabilistically based on the input value. This creates richer, value-dependent representations.
3. **Hybrid Transformer Module**: Augments the standard Transformer encoder layer with a parallel fully-connected aggregation path that exploits the fixed dimensionality of tabular data (unlike variable-length text/video).

The model is trained with a variational objective that includes KL divergence terms for the stochastic components, and inference uses **Bayesian averaging** over multiple forward passes (N=64 samples) to marginalize out the stochastic choices.

---

## 2. Architecture

### 2.1 Feature Embedding

**Categorical features**: Standard embedding lookup (same as FT-Transformer).

**Numerical features** — the **Embedding Mixture Layer** (novel):

For each numerical feature x_i, instead of a single linear embedding `h_i = w·x_i + b`, STab maintains J alternative linear projections `{(w_j, b_j)}_{j=1}^{J}` and selects one probabilistically:

```
f_emb(x_i) = x_i · w_j + b_j

where j ~ P(·|x_i, θ_w, θ_b)
      P(j|x_i) = softmax(x_i · θ_w + θ_b)
```

The selection is implemented via Gumbel-Softmax reparameterization during training (T=0.69) and near-deterministic at inference (T=0.01). This allows the model to learn **value-dependent embeddings** — different input ranges can activate different linear projections, effectively creating piecewise-linear embeddings.

**Intuition**: A feature like "age" might use one linear projection for ages 0–30, another for 30–60, and a third for 60+, capturing behavioral changes across value ranges without explicit binning.

### 2.2 Hybrid Transformer Layer

Each Transformer layer contains two parallel paths:

```
Input tokens (batch × features × d)
  ├── Path 1 (Standard Transformer):
  │     → LayerNorm → Multi-Head Self-Attention (with learned bias) + residual
  │     → LayerNorm → LWTA Feed-Forward + residual
  │
  └── Path 2 (Parallel Aggregation Module — novel):
        → LocalLinear: reproject each d-dim token to scalar → (batch × features)
        → LayerNorm → LWTA layer → Linear → (batch × 1 × d)
        → Add to CLS token representation (residual)
```

**Key differences from standard Transformer**:

1. **Attention bias**: A learnable bias matrix is added to the attention dot-product scores (shape: heads × features × features), exploiting the fixed feature ordering of tabular data.
2. **LWTA replaces ReLU/GELU**: All feed-forward layers use LWTA competition instead of deterministic activations.
3. **Parallel module**: The GlobalResnet module provides a static feature-aggregation pathway that complements the dynamic attention mechanism.

### 2.3 Self-Attention with Learned Bias

```
Attention(Q, K, V) = softmax(Q·K^T/√d + B) · V

where B ∈ ℝ^(heads × features × features) is a learned bias
```

This is a departure from standard Transformers where positional information comes from sinusoidal or learnable position embeddings. Since tabular features have a fixed, meaningful ordering, the bias directly encodes pairwise feature interaction priors.

### 2.4 Local Winner Takes All (LWTA) Activation

LWTA replaces deterministic activations in all feed-forward layers:

```
For each block k of U neurons:
  y_k = ξ_k ⊙ (W_k · x)
  
  where ξ_k ~ Gumbel-Softmax(W_k · x, T)  (one-hot winner indicator)
```

- During training: Gumbel-Softmax with T=0.69 (soft, differentiable)
- During inference: T=0.01 (near-hard selection)
- Block size U=2 (binary competition — each pair of neurons, one wins, one is zeroed)

**Effect**: Promotes sparse, diverse representations. Acts as a learned form of dropout — the network learns *which* neurons to activate per input, rather than dropping randomly.

### 2.5 Training Objective

The loss function is a variational objective:

```
L(φ) = E_q[log p(D|φ)] - KL[Q(ξ)||P(ξ)] - KL[Q(j)||P(j)]

where:
  - First term: standard task loss (cross-entropy or MSE)
  - Second term: KL divergence for LWTA winner indicators (uniform prior)
  - Third term: KL divergence for embedding selection indicators (uniform prior)
```

The KL terms act as regularizers, penalizing overconfident competition outcomes and encouraging exploration of different neurons/embeddings.

### 2.6 Bayesian Inference

At inference, the stochastic model produces different outputs for each forward pass (different LWTA winners, different embedding selections). The final prediction is the **average over N forward passes**:

```
ŷ = (1/N) Σ_{n=1}^{N} f(x; ξ_n, j_n)    where ξ_n, j_n are sampled per pass
```

Default: N=64 samples. Performance plateaus around N=20 but improves slightly up to 64.

**Key distinction from ensembling**: Only ONE model is trained. The N forward passes reuse the same weights but with different stochastic choices. No additional training cost — only inference cost scales linearly with N (and can be parallelized via batch replication on GPU).

---

## 3. Hyperparameters (Search Space in This Thesis)

| Parameter | Range | Scale | Effect |
|-----------|-------|-------|--------|
| `depth` | 2–7 | Linear | Number of Hybrid Transformer layers |
| `dim` | {16, 64, 96, 128, 192, 256} | Categorical | Token embedding dimension |
| `heads` | {4, 8} | Categorical | Attention heads (must divide dim) |
| `attn_dropout` | 0.0–0.3 | Linear | Dropout on attention weights (MC-dropout) |
| `ff_dropout` | 0.0–0.3 | Linear | Dropout in feed-forward layers (MC-dropout) |
| `cases` | {4, 8, 16} | Categorical | J: number of alternative embeddings in Mixture Layer |
| `lwta_block_size` | 2 | Fixed | U: LWTA block size (authors recommend fixed at 2) |
| `n_inference_samples` | 64 | Fixed | N: Bayesian averaging samples at inference |
| `learning_rate` | 1e-5–1e-3 | Log | AdamW learning rate |
| `weight_decay` | 1e-5–1e-3 | Log | AdamW weight decay |
| `kl_weight` | 0.01 | Fixed | Weight for KL divergence terms in loss |

**Paper-recommended defaults**: dim=256, depth=4, heads=8, U=2, J=16, dropout=0.25, T_train=0.69, T_infer=0.01, N=64, lr=1e-3, wd=1e-4

---

## 4. Why It Works

| Property | Benefit |
|----------|---------|
| **Stochastic activations (LWTA)** | Implicit regularization through learned sparsity — prevents overfitting without extra parameters |
| **Embedding Mixture** | Value-dependent feature representations — piecewise-linear embeddings capture non-linear feature effects |
| **Hybrid architecture** | Combines dynamic (attention) and static (FC aggregation) feature interactions — best of both worlds |
| **Attention bias** | Exploits fixed tabular feature ordering — learns pairwise feature interaction priors |
| **Bayesian averaging** | Multiple samples reduce prediction variance — implicit ensemble from a single model |
| **KL regularization** | Variational objective prevents competition collapse — maintains diversity in LWTA and embedding selection |

---

## 5. Strengths and Weaknesses

### Strengths
- **Built-in regularization**: LWTA + Embedding Mixture + KL divergence provide three layers of regularization, reducing overfitting on small-to-medium tabular datasets
- **Rich numerical embeddings**: The Embedding Mixture Layer creates more expressive feature representations than single linear projections, especially for features with complex value-dependent behavior
- **Bayesian uncertainty**: Multiple forward passes naturally yield prediction uncertainty estimates — valuable for real-world decision-making
- **Single-model ensemble effect**: N=64 Bayesian averaging samples at inference without N× training cost
- **Strong empirical results**: Achieves best or second-best performance on 5/8 benchmarks against FT-Transformer, MLP-PLR, NODE, and SAINT (single model); 5/8 including GBDT ensembles (ensemble setting)

### Weaknesses
- **Stochastic inference cost**: N=64 forward passes makes inference 64× slower than a single pass — problematic for latency-sensitive applications
- **Depends on keras4torch**: The reference implementation uses the `keras4torch` library for training, which is unmaintained and has limited community support
- **Experimental code quality**: The official repository is in "experimental version" (per the authors) — code quality is not production-grade
- **Limited to 8 benchmarks**: Validated on a narrower set of datasets compared to other models in this thesis
- **Categorical feature handling**: Performance is comparatively weaker on datasets with many categorical features (AD, DI) — the Embedding Mixture only benefits numerical features
- **Gumbel-Softmax sensitivity**: Temperature scheduling (T=0.69 train → T=0.01 inference) requires careful tuning; incorrect temperatures can cause training instability or poor convergence
- **Computational overhead from hybrid module**: ~28% more parameters and ~42% longer training time vs standard Transformer encoder (per the paper's ablation)

---

## 6. Relation to Other Models in This Thesis

| Model | Relationship to STab |
|-------|---------------------|
| **FT-Transformer** | STab's base architecture. STab = FT-Transformer + LWTA + Embedding Mixture + Hybrid Module. The ablation study shows the "Vanilla" variant (no additions) is exactly FT-Transformer |
| **SAINT** | Both extend FT-Transformer. SAINT adds inter-sample (row) attention; STab adds stochastic competition and hybrid aggregation. Different approaches to improving the Transformer for tabular data |
| **TabNet** | Both use attention + sparsity. TabNet uses sequential attention for feature selection; STab uses LWTA for sparse activation. TabNet is deterministic; STab is stochastic |
| **TabM** | Both provide implicit ensembling. TabM uses BatchEnsemble (K ensemble members with shared weights); STab uses Bayesian averaging over stochastic forward passes. Different diversity mechanisms |
| **MLP** | STab's parallel aggregation module is essentially an MLP path, consistent with the finding that MLPs remain competitive when properly regularized |
| **GBDTs** | STab achieves comparable or better results on most benchmarks. Weaker on datasets with categorical features where CatBoost excels |

---

## 7. Key References

| Reference | Contribution |
|-----------|--------------|
| Voskou, Christoforou & Chatzis (2024) | **STab** — the proposed model |
| Panousis et al. (2019) | LWTA — the stochastic competition activation used in STab |
| Gorishniy et al. (2021) | FT-Transformer — the base architecture STab builds upon |
| Jang et al. (2016) | Gumbel-Softmax — the reparameterization trick enabling differentiable discrete sampling |
| Voskou et al. (2021) | Stochastic Transformer Networks — predecessor work applying LWTA to Transformers for sign language |
| Gorishniy et al. (2022) | Numerical feature embeddings (PLR) — motivation for STab's Embedding Mixture Layer |

---

## 8. In This Thesis

STab tests whether **stochastic competition** and **architectural hybridization** can bridge the DL–GBDT gap on tabular data. It represents a fundamentally different regularization strategy from the other DL models in this benchmark:

- **FT-Transformer** relies on standard dropout
- **SAINT** relies on inter-sample attention for implicit regularization
- **TabM** relies on BatchEnsemble for diversity
- **TabNet** relies on sequential attention sparsity
- **STab** relies on *learned stochastic sparsity* (LWTA) and *Bayesian averaging*

This makes STab a unique data point in the analysis — if it outperforms, it suggests that stochastic regularization is more effective than architectural complexity for tabular data.

**Preprocessing**: Standardize numericals, ordinal-encode categoricals, median-impute. (Same DL preprocessing pipeline as FT-Transformer, SAINT, TabM.)

**Implementation note**: We implement STab as a self-contained PyTorch module (vendored), since the official repository relies on `keras4torch` which is not compatible with our unified training pipeline. The core architecture (LWTA, Embedding Mixture, Hybrid Transformer) is faithfully reimplemented from the paper and official source code.
