# TabM (Tabular Model with Batch Ensembling)

> **Paper**: Gorishniy et al., "TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling", ICLR 2025  
> **Role in this thesis**: A parameter-efficient ensemble of MLPs that achieves state-of-the-art deep learning performance on tabular data through batch ensembling — sharing most weights across ensemble members while using lightweight per-member adapters.

---

## 1. What It Is

TabM takes a simple but powerful approach: instead of a single MLP, it trains **K ensemble members simultaneously** using **Batch Ensemble** layers. The key insight is that most of the MLP weights can be **shared** across ensemble members, with only small per-member scaling vectors (adapters) that differentiate them.

This is fundamentally different from training K independent MLPs (which would be K× more expensive). BatchEnsemble shares >95% of parameters while achieving ensemble-level diversity.

---

## 2. Architecture

### 2.1 Batch Ensemble Linear Layer

A standard linear layer computes `y = Wx + b`. A BatchEnsemble layer computes:

```
For ensemble member k:
  y_k = (W ⊙ (rₖ · sₖᵀ)) · x + bₖ

Where:
  W ∈ ℝ^(out × in)     — shared weight matrix (same for all K members)
  rₖ ∈ ℝ^out            — per-member output scaling vector
  sₖ ∈ ℝ^in             — per-member input scaling vector  
  bₖ ∈ ℝ^out            — per-member bias
```

The rank-1 perturbation `rₖ · sₖᵀ` modifies the shared weights differently for each member, creating diversity with minimal extra parameters.

### 2.2 Full Architecture

```
Input x ∈ ℝⁿ
  → EnsembleView: replicate x into K copies
  → [BatchEnsemble Linear → BatchNorm → ReLU → Dropout] × n_blocks
  → LinearEnsemble: K independent output heads
  → K predictions
  → Average at inference time
```

### 2.3 Training

**Critical detail**: The loss is the **mean of per-member losses**, not the loss of the mean prediction:

```
Loss = (1/K) · Σₖ L(ŷₖ, y)    ← each member independently predicts and incurs loss
```

This encourages each member to be individually accurate, promoting meaningful diversity.

---

## 3. Hyperparameters (Search Space in This Thesis)

| Parameter | Range | Scale | Effect |
|-----------|-------|-------|--------|
| `arch_type` | {tabm, tabm-mini} | Categorical | Full vs lightweight variant |
| `k` | 8–64 | Linear | Number of ensemble members |
| `d_block` | {128, 256, 512} | Categorical | Hidden layer width |
| `n_blocks` | 1–4 | Linear | Number of MLP blocks |
| `dropout` | 0.0–0.5 | Linear | Dropout rate |
| `learning_rate` | 1e-5–1e-3 | Log | AdamW learning rate |
| `weight_decay` | 1e-6–1e-3 | Log | L2 regularization (AdamW decoupled weight decay) |

---

## 4. Why It Works

| Property | Benefit |
|----------|---------|
| **Ensemble diversity** | K different predictions from different weight perturbations reduce variance |
| **Parameter efficiency** | Only 2×(in+out) extra params per member per layer vs in×out for independent MLPs |
| **Implicit regularization** | Shared backbone prevents individual members from memorizing |
| **Single forward pass** | All K members computed simultaneously via batched operations — same cost as K× batch size |

---

## 5. Strengths and Weaknesses

### Strengths
- **State-of-the-art DL for tabular**: Matches or exceeds FT-Transformer on most benchmarks
- **Simple**: Just an MLP with batch ensembling — no attention, no complex architecture
- **Efficient**: K members at roughly the cost of one model (shared weights)
- **Well-regularized**: Ensembling + weight sharing + dropout provides strong regularization

### Weaknesses
- **Still an MLP at core**: No explicit feature interaction modeling (unlike attention-based methods)
- **Memory scales with K**: More ensemble members = larger batch dimension
- **New (limited ecosystem)**: Less battle-tested than XGBoost or standard MLPs

---

## 6. Key References

| Reference | Contribution |
|-----------|--------------|
| Wen et al. (2020) | BatchEnsemble — the weight-sharing ensemble technique |
| Gorishniy et al. (2024/2025) | **TabM** — applied BatchEnsemble to tabular MLPs |
| Gorishniy et al. (2021) | Revisiting DL for Tabular Data (same group, earlier work) |

---

## 7. In This Thesis

TabM represents the **modern MLP** approach: take the simplest architecture (MLP) and make it work better through efficient ensembling. It directly tests whether ensemble diversity beats architectural complexity (attention, feature selection).

**Preprocessing**: Standardize numericals, one-hot encode categoricals, median-impute.
