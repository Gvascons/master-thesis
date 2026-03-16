# XGBoost (eXtreme Gradient Boosting)

> **Paper**: Chen & Guestrin, "XGBoost: A Scalable Tree Boosting System", KDD 2016  
> **Role in this thesis**: Primary GBDT baseline — the model to beat. XGBoost popularized gradient boosting at scale and remains the default choice for tabular data in industry and Kaggle competitions.

---

## 1. What It Is

XGBoost is a **gradient-boosted decision tree (GBDT)** framework. It builds an ensemble of decision trees sequentially, where each new tree corrects the errors (residuals) of the previous ensemble. The key innovation over earlier GBDT implementations (e.g., Friedman's 2001 gradient boosting machine) is a combination of **regularized learning objective**, **second-order gradient approximation**, and **systems-level optimizations** that made it orders of magnitude faster and more scalable.

Before XGBoost, GBDTs already dominated structured/tabular data competitions. XGBoost made them practical at scale — handling millions of rows and sparse features efficiently — and became the most widely used ML algorithm for tabular prediction from 2014 to roughly 2020.

---

## 2. Core Concept: Gradient Boosting

### 2.1 Boosting Framework

Gradient boosting builds a prediction model as an **additive ensemble** of weak learners (shallow decision trees):

```
ŷᵢ⁽⁰⁾ = 0                              (initial prediction)
ŷᵢ⁽¹⁾ = ŷᵢ⁽⁰⁾ + η · f₁(xᵢ)             (add tree 1)
ŷᵢ⁽²⁾ = ŷᵢ⁽¹⁾ + η · f₂(xᵢ)             (add tree 2)
...
ŷᵢ⁽ᵗ⁾ = Σₖ₌₁ᵗ η · fₖ(xᵢ)              (final: sum of T trees)
```

Where:
- `fₖ` is the k-th decision tree
- `η` is the **learning rate** (shrinkage) — scales each tree's contribution to prevent overfitting
- Each tree `fₖ` is fit to the **negative gradient** of the loss w.r.t. the current prediction

### 2.2 Why "Gradient" Boosting?

Each tree doesn't fit the original labels — it fits the gradient of the loss function:

```
For regression (MSE loss):
  gradient = -(yᵢ - ŷᵢ⁽ᵗ⁻¹⁾) = residual
  → Each tree literally predicts the "mistake" of the current ensemble

For classification (log-loss):
  gradient = -(yᵢ - pᵢ⁽ᵗ⁻¹⁾)
  → Each tree corrects the probability prediction toward the true class
```

This is the fundamental insight: gradient boosting converts any differentiable loss function into a tree-fitting problem by chasing the loss gradient.

---

## 3. XGBoost's Innovations

### 3.1 Regularized Objective

XGBoost adds explicit regularization to the tree-building objective, unlike classic GBDT:

```
Obj⁽ᵗ⁾ = Σᵢ L(yᵢ, ŷᵢ⁽ᵗ⁻¹⁾ + fₜ(xᵢ)) + γ·T + ½·λ·Σⱼ wⱼ²
```

Where:
- `L` = loss function (e.g., log-loss, MSE)
- `γ·T` = penalty on the number of leaves `T` (controls tree complexity)
- `½·λ·Σⱼ wⱼ²` = L2 penalty on leaf weights `wⱼ` (prevents extreme predictions)

This is a critical difference from earlier GBDT — regularization is baked into the split-finding algorithm, not applied post-hoc.

### 3.2 Second-Order Taylor Approximation

Classic GBDT uses only the first derivative (gradient) of the loss. XGBoost uses a **second-order Taylor expansion**, incorporating the Hessian (curvature):

```
Obj⁽ᵗ⁾ ≈ Σᵢ [gᵢ · fₜ(xᵢ) + ½ · hᵢ · fₜ(xᵢ)²] + Ω(fₜ)

Where:
  gᵢ = ∂L/∂ŷ⁽ᵗ⁻¹⁾       (gradient — direction of error)
  hᵢ = ∂²L/∂(ŷ⁽ᵗ⁻¹⁾)²   (hessian — curvature of error)
```

Using both gradient and hessian provides a more precise approximation, leading to better split decisions and optimal leaf weights:

```
Optimal leaf weight:  w*ⱼ = -Σᵢ∈leaf gᵢ / (Σᵢ∈leaf hᵢ + λ)
Split gain:           Gain = ½ · [G²_L/(H_L+λ) + G²_R/(H_R+λ) - (G_L+G_R)²/(H_L+H_R+λ)] - γ
```

### 3.3 Split-Finding Algorithm

For each candidate split, XGBoost computes the gain formula above. The split with the highest gain is chosen. Key efficiency tricks:

- **Pre-sorted algorithm**: Sort features once, scan sorted values for best split. O(n·d·K) for n samples, d features, K trees.
- **Histogram-based approximation**: Bucket continuous features into quantiles, reducing candidate splits. Enabled via `tree_method='hist'`.
- **Sparsity-aware**: Natively handles missing values by learning a default direction at each split node — missing values go left or right based on which improves the objective more.

### 3.4 Systems-Level Optimizations

| Optimization | Description |
|-------------|-------------|
| **Column block** | Data stored in compressed column format for cache-efficient access during split finding |
| **Parallelization** | Feature-level parallelism — multiple features evaluated simultaneously across CPU cores |
| **Cache-aware access** | Buffer-based gradient accumulation to reduce cache misses |
| **Out-of-core** | Block-based external memory for datasets too large for RAM |

These aren't algorithm changes — they're engineering that makes XGBoost 10× faster than equivalent implementations.

---

## 4. Decision Tree Mechanics

Each tree in the ensemble is a **CART (Classification and Regression Tree)**:

```
         [feature_5 < 0.73?]
         /                 \
    Yes (≤)              No (>)
       /                    \
  [feature_2 < 12?]     leaf: w = -0.04
    /           \
leaf: w = 0.15  leaf: w = -0.08
```

- **Internal nodes**: axis-aligned splits on a single feature (e.g., `age < 45`)
- **Leaf nodes**: contain a scalar weight `w` — the tree's prediction for samples reaching that leaf
- **Max depth** controls the maximum number of sequential splits (default search: 3–12)

Shallow trees (depth 3–6) are typical. Each individual tree is a "weak learner" — barely better than random. The ensemble's power comes from combining hundreds to thousands of these corrections.

---

## 5. Training Procedure in This Thesis

### 5.1 Early Stopping

XGBoost trains up to `n_estimators` trees (search range: 100–2000), but stops early if the evaluation metric on the validation set doesn't improve for `early_stopping_rounds` (default: 50) consecutive rounds. This is analogous to early stopping in neural networks — it prevents overfitting by selecting the optimal ensemble size.

### 5.2 Stochastic Components

Two forms of randomization reduce overfitting and decorrelate trees:

| Parameter | Range | Effect |
|-----------|-------|--------|
| `subsample` | 0.5–1.0 | Row sampling — each tree sees a random fraction of the training data |
| `colsample_bytree` | 0.5–1.0 | Column sampling — each tree considers a random subset of features |

Both are forms of **bagging** applied within the boosting framework. They reduce variance and speed up training.

---

## 6. Hyperparameters (Search Space in This Thesis)

| Parameter | Range | Scale | Effect |
|-----------|-------|-------|--------|
| `n_estimators` | 100–2000 | Linear | Number of trees. More trees = more capacity, but slower + early stopping decides |
| `max_depth` | 3–12 | Linear | Maximum tree depth. Deeper trees capture higher-order feature interactions but overfit faster |
| `learning_rate` | 0.005–0.3 | Log | Shrinkage. Lower values need more trees but generalize better |
| `subsample` | 0.5–1.0 | Linear | Row sampling ratio per tree |
| `colsample_bytree` | 0.5–1.0 | Linear | Feature sampling ratio per tree |
| `min_child_weight` | 1.0–10.0 | Linear | Minimum sum of hessians in a leaf. Higher = more conservative (regularization) |
| `reg_alpha` | 1e-8–10.0 | Log | L1 regularization on leaf weights (promotes sparsity) |
| `reg_lambda` | 1e-8–10.0 | Log | L2 regularization on leaf weights (the λ in the regularized objective) |

The interplay between `learning_rate` and `n_estimators` is critical: smaller learning rates need more trees, and the optimal combination is found via early stopping.

---

## 7. Why It Dominates Tabular Data

### Strengths

- **Axis-aligned splits are natural for tabular data**: Unlike images where pixels have spatial relationships, tabular features are independent columns. Tree splits (`feature < threshold`) are the natural inductive bias.
- **Native handling of heterogeneous features**: Mixes numerical and categorical features without preprocessing (ordinal encoding suffices).
- **Missing value support**: Learned default direction per split — no imputation needed (though we impute for consistency across model families).
- **Feature interactions**: A tree of depth `d` captures `d`-way feature interactions automatically. A depth-6 tree can model 6-way interactions.
- **Robustness to scale**: Trees are invariant to monotone transformations (log, scaling) — no need to standardize features.
- **Fast training and inference**: Orders of magnitude faster than neural networks for equivalent accuracy.
- **Interpretable**: Feature importance, SHAP values, and tree visualization provide explainability.

### Weaknesses

- **Axis-aligned only**: Splits are perpendicular to feature axes. Diagonal decision boundaries require many splits to approximate (staircase pattern).
- **Extrapolation**: Trees can only predict values within the range seen during training. For regression, they cannot extrapolate beyond the training distribution.
- **High cardinality categoricals**: Without target encoding, high-cardinality categoricals create sparse splits. CatBoost addresses this directly.
- **No representation learning**: Each tree operates on raw features. Neural networks learn latent representations that may capture abstract patterns.
- **Diminishing returns from depth**: Very deep trees overfit rapidly. Complex feature interactions are better captured by more trees than deeper trees.

---

## 8. Key References

| Reference | Contribution |
|-----------|--------------|
| Breiman et al. (1984) | CART — the foundation for decision tree mechanics |
| Freund & Schapire (1997) | AdaBoost — the precursor boosting algorithm |
| Friedman (2001) | Gradient Boosting Machine — generalized boosting to any differentiable loss |
| Chen & Guestrin (2016) | **XGBoost** — regularized objective, 2nd-order approximation, systems engineering |
| Grinsztajn et al. (2022) | "Why do tree-based models still outperform deep learning on typical tabular data?" — theoretical grounding |

---

## 9. In This Thesis

XGBoost is part of the **GBDT family** alongside LightGBM and CatBoost. Together, they represent the current state-of-the-art on tabular data. The key research question is whether deep learning or foundation models (TabPFN) can match or surpass this family.

**Preprocessing**: Ordinal-encode categoricals, median-impute missing values. No standardization needed (trees are scale-invariant).

XGBoost specifically serves as the "original" GBDT baseline — LightGBM and CatBoost are its direct descendants with specific engineering and algorithmic improvements.
