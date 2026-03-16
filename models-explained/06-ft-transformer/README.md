# FT-Transformer (Feature Tokenizer + Transformer)

> **Paper**: Gorishniy et al., "Revisiting Deep Learning Models for Tabular Data", NeurIPS 2021  
> **Role in this thesis**: The first successful application of the Transformer architecture to tabular data. FT-Transformer treats each feature as a "token" and applies self-attention to learn pairwise feature interactions.

---

## 1. What It Is

FT-Transformer adapts the Transformer architecture (Vaswani et al., 2017) — originally designed for sequences (text, speech) — to tabular data. The key insight: **treat each feature as a token**. Where BERT tokenizes words, FT-Transformer tokenizes features.

Each feature (e.g., "age", "income", "education") is projected into a d-dimensional embedding space, then processed through standard Transformer blocks with multi-head self-attention. This allows the model to learn **pairwise feature interactions** directly — feature A attending to feature B captures their joint effect.

This paper is one of the most cited in tabular deep learning because it also provided a rigorous benchmark showing that:
1. A well-tuned MLP is already a strong baseline
2. FT-Transformer is the best DL model across diverse tabular benchmarks
3. GBDTs still win overall, but the gap is narrowing

---

## 2. Architecture

### 2.1 Feature Tokenizer

Each numerical feature `xⱼ` is projected into a d-dimensional token:

```
For each feature j:
  tokenⱼ = xⱼ · wⱼ + bⱼ     (element-wise: scalar → d-dim vector)
```

Where `wⱼ ∈ ℝᵈ` and `bⱼ ∈ ℝᵈ` are learnable per-feature projection parameters.

A special **[CLS] token** is prepended (like BERT):
```
Tokens: [CLS, token₁, token₂, ..., tokenₙ]
        ↑ this is the "summary" token used for prediction
```

### 2.2 Transformer Block (× n_blocks)

Each Transformer block applies:

```
z = LayerNorm(tokens)
z = MultiHeadSelfAttention(z) + tokens     ← residual connection
z = LayerNorm(z)
z = FFN(z) + z                              ← residual connection
```

#### Multi-Head Self-Attention

Attention computes pairwise interactions between all features:

```
Attention(Q, K, V) = softmax(QKᵀ / √d_k) · V

Where:
  Q = z · W_Q    (queries — "what am I looking for?")
  K = z · W_K    (keys — "what do I offer?")
  V = z · W_V    (values — "what information do I carry?")
```

Multi-head = run attention `h` times with different projections, then concatenate:
```
MultiHead(z) = Concat(head₁, ..., headₕ) · W_O
```

**Why this matters for tabular data**: Each feature "attends" to every other feature. The attention weights `softmax(QKᵀ/√d_k)` reveal which feature pairs interact. For example, "age" might strongly attend to "income" (age-income interaction) while ignoring "zip_code".

#### Feed-Forward Network (FFN)

After attention, each token is processed independently. The implementation uses **ReGLU** (ReLU-Gated Linear Unit), a GLU variant:

```
ReGLU(x) = x₁ ⊙ ReLU(x₂)    (split input, gate with ReLU)
FFN(z) = Dropout(ReGLU(z · W₁ + b₁)) · W₂ + b₂
```

ReGLU provides gating (like GLU in TabNet) but with ReLU instead of sigmoid. The paper shows this outperforms standard ReLU or GELU activations in the FFN.

The FFN width is controlled by `ffn_d_hidden_multiplier`: inner dimension = `d_block × multiplier`. The multiplier range (1.33–2.67) controls the expansion ratio.

### 2.3 Prediction Head

After all Transformer blocks, the **[CLS] token** is extracted and projected to the output:
```
output = Linear(CLS_token_final)
```

The [CLS] token aggregates information from all features through attention across all layers.

---

## 3. Why Transformers for Tabular Data?

### The Feature Interaction Argument

| Model | How it captures feature interactions |
|-------|--------------------------------------|
| **MLP** | Implicitly, through stacked linear+nonlinear layers. Must discover interactions from scratch. |
| **GBDT** | One feature per split. Tree depth `d` → up to `d`-way interactions, but only axis-aligned. |
| **FT-Transformer** | Explicitly via attention. Every pair of features is compared in every layer. |

FT-Transformer's attention mechanism provides a natural way to model **all pairwise** feature interactions simultaneously. This is its core advantage over MLPs (which must learn interactions indirectly) and GBDTs (which only model one feature at a time per split).

### The Permutation Invariance Argument

Tabular features have no natural ordering (unlike words in a sentence or pixels in an image). Transformers with self-attention are **permutation equivariant** — reordering the features doesn't change the output (up to the [CLS] token). This is a desirable property for tabular data.

Note: FT-Transformer doesn't use positional encodings (unlike NLP Transformers), precisely because feature order is arbitrary.

---

## 4. Hyperparameters (Search Space in This Thesis)

| Parameter | Range | Scale | Effect |
|-----------|-------|-------|--------|
| `n_blocks` | 1–6 | Linear | Transformer depth. More blocks = higher-order interactions |
| `d_block` | {64, 128, 192, 256} | Categorical | Token/embedding dimension |
| `attention_n_heads` | {4, 8} | Categorical | Number of attention heads |
| `attention_dropout` | 0.0–0.5 | Linear | Dropout on attention weights |
| `ffn_d_hidden_multiplier` | 1.33–2.67 | Linear | FFN expansion ratio |
| `ffn_dropout` | 0.0–0.5 | Linear | Dropout in FFN |
| `residual_dropout` | 0.0–0.2 | Linear | Dropout on residual connections |
| `learning_rate` | 1e-5–1e-3 | Log | AdamW learning rate |
| `weight_decay` | 1e-6–1e-3 | Log | L2 regularization |

---

## 5. Strengths and Weaknesses

### Strengths

- **Explicit feature interactions**: Attention directly models pairwise feature relationships
- **Flexible capacity**: Depth and width are independently controllable
- **Proven architecture**: Transformers are the dominant architecture in NLP/vision — well-studied optimization landscape
- **Feature importance**: Attention weights provide interpretability (which features interact)
- **Best DL model in original benchmark**: Outperformed MLP, TabNet, and other DL approaches

### Weaknesses

- **Quadratic complexity**: Self-attention is O(n²) in the number of features. For wide datasets (100+ features), this becomes expensive
- **Data hungry**: Needs more data than GBDTs to learn meaningful attention patterns
- **Slower than GBDTs**: Matrix multiplications and attention computation add overhead
- **Not always better than MLP**: On some datasets, the added complexity of attention doesn't help
- **Pretraining not leveraged**: Unlike NLP Transformers, there's no large-scale pretraining benefit (until TabPFN)

---

## 6. Key References

| Reference | Contribution |
|-----------|--------------|
| Vaswani et al. (2017) | "Attention Is All You Need" — the Transformer architecture |
| Devlin et al. (2019) | BERT — [CLS] token idea, pre-training then fine-tuning |
| Gorishniy et al. (2021) | **FT-Transformer** — adapted Transformer for tabular data with feature tokenization |
| Huang et al. (2020) | TabTransformer — earlier work on transformers for tabular (categoricals only) |

---

## 7. In This Thesis

FT-Transformer represents the **Transformer approach** to tabular deep learning. Key comparisons:
- vs. MLP: Does explicit attention over features beat implicit interaction learning?
- vs. GBDTs: Can learned pairwise interactions compete with axis-aligned splits?
- vs. TabNet: Pairwise attention (all features attend to all) vs. sequential feature selection

**Preprocessing**: Standardize numericals, one-hot encode categoricals, median-impute — standard DL pipeline. All features treated as continuous tokens.

**Implementation note**: Uses the `rtdl_revisiting_models` library for the Transformer backbone. The wrapper provides a custom training loop with early stopping.
