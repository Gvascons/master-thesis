# RealMLP (Realistic MLP Baseline)

> **Paper**: Holzmüller, Grinsztajn & Steinwart, "Better by Default: Strong Pre-Tuned MLPs and Boosted Trees on Tabular Data", NeurIPS 2024  
> **Role in this thesis**: A carefully engineered MLP that achieves strong performance through smart defaults and preprocessing rather than architectural novelty. Top performer in the TabArena benchmark.

---

## 1. What It Is

RealMLP is not a new architecture — it's a standard MLP made competitive through **engineering excellence**:

1. **Neural Tangent Parameterization** — weight initialization that stabilizes training at any depth
2. **Parametric Mish activation** — learnable activation functions that adapt per layer
3. **Robust preprocessing** — smooth clipping, robust scaling, piecewise-linear numerical embeddings
4. **Carefully tuned defaults** — hyperparameters pre-optimized across hundreds of datasets

The thesis of the paper: most "novel" tabular DL architectures derive their gains from **better training recipes**, not from architectural innovations. When you give an MLP the same treatment, it matches them.

---

## 2. Key Innovations

### 2.1 Neural Tangent Parameterization

Standard initialization (e.g., Kaiming, Xavier) becomes unstable for deeper networks. RealMLP uses NTK-inspired parameterization:

```
y = (1/√width) · W · x    (scale output by 1/√width)
```

This ensures that the output variance stays bounded regardless of network depth, enabling stable training of deeper MLPs without careful learning rate tuning.

### 2.2 Parametric Mish Activation

Instead of fixed ReLU or GELU, RealMLP uses Mish with a learnable parameter:

```
Mish(x) = x · tanh(softplus(x))
Parametric Mish(x; β) = x · tanh(softplus(β · x))
```

Where `β` is learned per-layer. This allows the activation shape to adapt: sharper (more ReLU-like) or smoother (more linear) depending on what the data needs.

### 2.3 Robust Preprocessing

- **Smooth clipping**: Clips extreme values using a smooth function instead of hard cutoffs, preserving gradient flow
- **Robust scaling**: Uses median and IQR instead of mean and std, reducing sensitivity to outliers
- **Piecewise-linear numerical embeddings**: Maps each numerical feature through a learned piecewise-linear function (like a 1D lookup table), capturing non-linear relationships before the MLP

### 2.4 Front Scale Layer

A learnable scaling layer applied to the input before the MLP, allowing the network to learn per-feature importance before processing.

---

## 3. Hyperparameters (Search Space in This Thesis)

| Parameter | Range | Scale | Effect |
|-----------|-------|-------|--------|
| `learning_rate` | 0.02–0.3 | Log | Much higher than standard MLPs (enabled by NTK parameterization) |
| `hidden_width` | {64, 256, 512} | Categorical | Width of hidden layers |
| `n_hidden_layers` | 1–5 | Linear | Depth |
| `dropout` | {0.0, 0.15, 0.3} | Categorical | Dropout rate |
| `weight_decay` | {0.0, 0.02} | Categorical | L2 regularization |
| `activation` | {mish, relu, selu} | Categorical | Activation function |
| `num_emb_type` | {none, pbld, pl, plr} | Categorical | Numerical embedding type: none (raw), piecewise-linear (pl), or with learnable breakpoints (pbld/plr) |
| `add_front_scale` | {true, false} | Categorical | Whether to add the learnable front scale layer |
| `ls_eps` | {0.0, 0.1} | Categorical | Label smoothing epsilon (0 = off, 0.1 = moderate smoothing) |

Note the significantly **higher learning rates** (0.02–0.3 vs 1e-5–1e-3 for standard MLPs). This is enabled by the NTK parameterization which stabilizes training at higher LRs.

---

## 4. Strengths and Weaknesses

### Strengths
- **Simple yet competitive**: No attention, no ensembling — just a well-trained MLP
- **Strong defaults**: Works well out-of-box across diverse datasets
- **Fast**: Simpler than TabM, FT-Transformer, or SAINT
- **Robust preprocessing**: Handles outliers and non-linearities before the network
- **Top TabArena performer**: Validated across hundreds of datasets

### Weaknesses
- **Still an MLP**: No explicit feature interaction modeling
- **Less flexible**: The engineering choices (NTK param, robust scaling) are opinionated — may not suit all data distributions
- **Preprocessing dependency**: Performance relies heavily on the preprocessing pipeline

---

## 5. Key References

| Reference | Contribution |
|-----------|--------------|
| Holzmüller et al. (2024) | **RealMLP** — engineered MLP with NTK param and robust preprocessing |
| Jacot et al. (2018) | Neural Tangent Kernel theory (foundation for NTK parameterization) |
| Misra (2020) | Mish activation function |
| Grinsztajn et al. (2022) | "Why do tree-based models still outperform DL on tabular data?" (motivation) |

---

## 6. In This Thesis

RealMLP tests the hypothesis that **engineering and defaults beat architecture**. It's the most direct comparison:
- vs. MLP: Same architecture, different training recipe. How much do good defaults matter?
- vs. FT-Transformer/SAINT: Simple MLP vs complex attention. Does engineering trump architecture?
- vs. GBDTs: Can a well-tuned MLP actually close the gap?

**Preprocessing**: Handled internally by `pytabkit` — robust scaling, smooth clipping, numerical embeddings.

**Implementation note**: Uses `pytabkit.RealMLP_TD_Classifier`/`Regressor` with sklearn-like API.
