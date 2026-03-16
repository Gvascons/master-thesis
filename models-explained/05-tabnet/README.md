# TabNet (Attentive Interpretable Tabular Learning)

> **Paper**: Arik & Pfister, "TabNet: Attentive Interpretable Tabular Learning", AAAI 2021 (Google Research)  
> **Role in this thesis**: First attention-based deep learning model purpose-built for tabular data. TabNet introduced sequential attention for feature selection, giving neural networks an inductive bias closer to decision trees.

---

## 1. What It Is

TabNet is a deep learning architecture designed specifically for tabular data. Its key insight: rather than processing all features simultaneously (like an MLP), TabNet processes features in **sequential decision steps**, selecting a different subset of features at each step using **learned attention masks**. This mimics how decision trees select features at each split — but in a differentiable, end-to-end trainable way.

Key properties:
- **Instance-wise feature selection**: Different samples can use different features at each step
- **Sequential multi-step processing**: Each step refines the prediction and selects new features
- **Interpretability**: The attention masks reveal which features mattered for each prediction
- **Sparse feature selection**: Regularization encourages the model to use few features per step (like a tree selecting one feature per split)

---

## 2. Architecture

### 2.1 High-Level Flow

```
Input features (x)
  → Step 1: Select features via attention → Process selected features → Partial output
  → Step 2: Select NEW features via attention → Process → Partial output
  → ...
  → Step N: Select features → Process → Partial output
  → Sum all partial outputs → Final prediction
```

Each step contributes a partial prediction, and the final output is the **sum** of all partial outputs (similar to how boosting sums tree predictions).

### 2.2 Core Components

#### Attentive Transformer (Feature Selection)

At each step `i`, an attention mechanism selects which features to process:

```
Mask_i = sparsemax(P_i · h(a_{i-1}))
```

Where:
- `a_{i-1}` is the processed information from the previous step
- `h(·)` is a fully-connected layer (the "prior scale" network)
- `P_i` is a **prior scale** that penalizes re-selecting features already used in previous steps
- `sparsemax` (or softmax) produces a sparse attention mask — most entries are exactly zero

The **prior scale** `P_i` is crucial:
```
P_i = Π_{j<i} (γ - M_j)
```

Where `γ` (gamma) controls how much feature reuse is allowed:
- `γ = 1.0`: Each feature can only be used once across all steps (maximum diversity)
- `γ = 2.0`: Features can be reused freely
- The search range (1.0–2.0) controls this tradeoff

#### Feature Transformer (Processing)

Selected features are processed through shared and step-specific layers:

```
Selected features: x̃_i = Mask_i ⊙ x    (element-wise multiplication)
                     ↓
Shared layers: FC → BN → GLU            (shared across all steps)
                     ↓
Step-specific layers: FC → BN → GLU      (unique to step i)
                     ↓
Split → [decision output] + [attention input for next step]
```

**GLU (Gated Linear Unit)**: `GLU(x) = x₁ ⊙ σ(x₂)` — splits the input in half, one half gates the other. This enables the network to learn which features to "let through" and which to suppress.

The output at each step splits into:
- `d_out` dimensions that contribute to the final prediction
- `d_a` dimensions that inform the next step's attention mechanism

#### Aggregation

```
Final output = Σᵢ ReLU(d_out_i)
```

All step outputs are summed (like gradient boosting) and passed through a final linear layer for prediction.

### 2.3 Sparsemax vs. Softmax

TabNet uses **sparsemax** (default) instead of softmax for attention:

| | Softmax | Sparsemax |
|---|---------|-----------|
| Output range | (0, 1) — all positive | [0, 1] — exactly zero allowed |
| Sparsity | Approximate (small values) | Exact (many zeros) |
| Interpretation | "All features matter a bit" | "Only these features matter" |
| Gradient | Always non-zero | Zero for unselected features |

Sparsemax produces truly sparse masks — most features get exactly zero attention, making TabNet's feature selection crisp and interpretable.

---

## 3. The TabNet–Decision Tree Connection

TabNet was explicitly designed to bridge the gap between neural networks and decision trees:

| Property | Decision Tree | TabNet |
|----------|--------------|--------|
| Feature selection | One feature per split | Sparse subset per step |
| Sequential processing | Split → left/right → split | Step → attend → process → step |
| Feature reuse | Features can be reused on different branches | Controlled by γ parameter |
| Ensemble output | Sum of leaf weights | Sum of step outputs |
| Interpretability | Feature importance from splits | Feature importance from attention masks |

---

## 4. Hyperparameters (Search Space in This Thesis)

| Parameter | Range | Scale | Effect |
|-----------|-------|-------|--------|
| `n_d` | {8, 16, 32, 64} | Categorical | Width of the decision output at each step |
| `n_a` | {8, 16, 32, 64} | Categorical | Width of the attention output at each step |
| `n_steps` | 3–10 | Linear | Number of sequential attention steps (like tree depth) |
| `gamma` | 1.0–2.0 | Linear | Feature reuse coefficient. 1 = no reuse, 2 = free reuse |
| `lambda_sparse` | 1e-6–1e-3 | Log | Sparsity regularization on attention masks |
| `learning_rate` | 0.005–0.05 | Log | Step size for optimization |
| `mask_type` | {sparsemax, entmax} | Categorical | Attention normalization function |

The `n_d`/`n_a` split controls the balance between decision capacity and attention capacity at each step. They're often set equal.

---

## 5. Strengths and Weaknesses

### Strengths

- **Built-in feature selection**: Learns which features matter per sample — no external feature selection needed
- **Interpretability**: Attention masks provide instance-level and global feature importance
- **Sparse processing**: Only processes relevant features per step — computationally efficient for wide datasets
- **Self-supervised pretraining**: TabNet supports a pretraining phase on unlabeled data (not used in this thesis)
- **No feature engineering**: End-to-end learning from raw features

### Weaknesses

- **Training instability**: Sensitive to learning rate and batch size. Can diverge or produce NaN losses
- **Slow convergence**: Needs many epochs and careful scheduling compared to GBDTs
- **Hyperparameter sensitive**: Performance varies significantly with `n_d`, `n_a`, `n_steps`, and `gamma`
- **Underperforms on small datasets**: The attention mechanism needs sufficient data to learn meaningful masks
- **Doesn't consistently beat GBDTs**: Despite the tree-like inductive bias, empirical results are mixed (Gorishniy et al., 2021)

---

## 6. Key References

| Reference | Contribution |
|-----------|--------------|
| Arik & Pfister (2021) | **TabNet** — sequential attention for tabular feature selection |
| Martins & Astudillo (2016) | Sparsemax — the sparse attention function TabNet relies on |
| Dauphin et al. (2017) | Gated Linear Units (GLU) — used in TabNet's feature transformer |
| Gorishniy et al. (2021) | "Revisiting DL for Tabular Data" — showed TabNet often underperforms simpler models |

---

## 7. In This Thesis

TabNet represents the **attention-based approach** to tabular deep learning. It's tested to determine whether its explicit feature selection mechanism provides an advantage over:
- MLPs (which process all features equally)
- GBDTs (which select features via greedy splits)
- Transformers (which compute pairwise feature attention)

**Preprocessing**: Standardize numericals, one-hot encode categoricals, median-impute — standard DL pipeline.

**Implementation note**: Uses the `pytorch-tabnet` library's sklearn-like API. The wrapper handles float32 conversion and reshaping for regression tasks.
