# LightGBM (Light Gradient Boosting Machine)

> **Paper**: Ke et al., "LightGBM: A Highly Efficient Gradient Boosting Decision Tree", NeurIPS 2017  
> **Role in this thesis**: Speed-optimized GBDT. LightGBM introduced histogram-based splitting and leaf-wise growth that made gradient boosting dramatically faster without sacrificing accuracy — often improving it.

---

## 1. What It Is

LightGBM is a gradient-boosted decision tree framework from Microsoft Research. Like XGBoost, it builds an additive ensemble of decision trees where each tree corrects the residuals of the current ensemble. The distinction is **how** trees are built:

- **Histogram-based split finding** instead of exact sorted splits
- **Leaf-wise (best-first) tree growth** instead of level-wise (depth-first)
- **Gradient-based One-Side Sampling (GOSS)** to reduce dataset size
- **Exclusive Feature Bundling (EFB)** to reduce feature dimensionality

These innovations made LightGBM 10–20× faster than XGBoost on large datasets while matching or exceeding its accuracy. It quickly became the default GBDT in production systems and Kaggle competitions.

---

## 2. Key Innovation #1: Histogram-Based Split Finding

### The Problem with XGBoost's Approach

XGBoost's pre-sorted algorithm evaluates **every unique feature value** as a candidate split point. For a dataset with `n` rows and `d` features, this is `O(n·d)` per tree level — expensive when `n` is large.

### LightGBM's Solution

LightGBM **buckets** continuous feature values into a fixed number of discrete bins (default: 255 bins):

```
Raw feature values:   [0.1, 0.3, 0.7, 0.2, 0.9, 0.5, ...]
                           ↓ histogram binning
Binned values:        [bin_0, bin_1, bin_3, bin_1, bin_4, bin_2, ...]
```

Split finding then scans **bins** instead of individual values:
- **Cost**: `O(bins·d)` instead of `O(n·d)` — with 255 bins, this is a massive speedup for large `n`
- **Memory**: 1 byte per feature per sample (bin index) instead of 4–8 bytes (float)
- **Cache efficiency**: Smaller data structures fit in CPU cache lines

### Histogram Subtraction Trick

When splitting a node, LightGBM only computes the histogram for the **smaller** child, then obtains the other child's histogram by subtracting from the parent:

```
histogram(right_child) = histogram(parent) - histogram(left_child)
```

This halves the histogram computation cost, a deceptively simple optimization with large practical impact.

---

## 3. Key Innovation #2: Leaf-Wise Tree Growth

### XGBoost: Level-Wise (Balanced Growth)

XGBoost grows trees level by level — all nodes at the same depth are split before moving deeper. This produces balanced trees but wastes splits on nodes with small gains.

```
Level-wise:
      [split]
     /       \
  [split]   [split]      ← all nodes at depth 1 are split
  /  \       /  \
 □    □     □    □        ← all at depth 2, even if gain is tiny
```

### LightGBM: Leaf-Wise (Best-First Growth)

LightGBM always splits the leaf with the **highest gain**, regardless of depth. This produces asymmetric trees that allocate splits where they matter most:

```
Leaf-wise:
         [split]
        /       \
     [split]     □           ← only the better side is split
    /       \
  [split]    □               ← splits go where gain is highest
  /     \
 □       □
```

**Benefit**: For the same number of leaves, leaf-wise growth achieves lower loss than level-wise. It converges faster — fewer trees needed for the same accuracy.

**Risk**: Leaf-wise can overfit on small datasets because it creates deeper, more asymmetric trees. The `num_leaves` parameter is the primary regularizer — it caps the maximum number of leaves per tree.

### num_leaves vs. max_depth

In LightGBM, `num_leaves` is more important than `max_depth`:
- A tree with `num_leaves = 31` and no depth limit can be very deep on one branch while shallow on others
- `max_depth` provides a secondary constraint to prevent extreme asymmetry
- Rule of thumb: `num_leaves` < 2^(`max_depth`) to avoid overfitting

---

## 4. Key Innovation #3: GOSS (Gradient-based One-Side Sampling)

Standard stochastic gradient boosting (subsample) randomly samples rows. GOSS is smarter:

```
1. Sort all samples by |gradient| (absolute value)
2. Keep the top a% (large gradient → poorly predicted → most informative)
3. Randomly sample b% from the remaining (small gradient → well predicted)
4. Upweight the sampled small-gradient instances by (1-a)/b to correct the distribution
```

**Intuition**: Samples with large gradients are the "hard" examples that the current ensemble gets wrong. They contribute most to the information gain. Samples with small gradients are already well-predicted and carry less information — we can subsample them without losing much.

This reduces the effective dataset size from `n` to `a·n + b·n` while preserving the gradient distribution.

---

## 5. Key Innovation #4: EFB (Exclusive Feature Bundling)

Many real-world datasets have sparse features (e.g., one-hot encoded categoricals) where features rarely take non-zero values simultaneously. EFB bundles these **mutually exclusive** features:

```
Feature A: [1, 0, 0, 1, 0, 0, 1, ...]    (sparse)
Feature B: [0, 1, 0, 0, 1, 0, 0, ...]    (sparse)
Feature C: [0, 0, 1, 0, 0, 1, 0, ...]    (sparse)
                    ↓ bundle
Bundle:    [1, 2, 3, 1, 2, 3, 1, ...]     (one feature with offset encoding)
```

This reduces `d` features to fewer bundles, speeding up histogram construction proportionally. The bundling is found via a graph-coloring algorithm (features that co-occur are different "colors" and can't be bundled).

---

## 6. Hyperparameters (Search Space in This Thesis)

| Parameter | Range | Scale | Effect |
|-----------|-------|-------|--------|
| `n_estimators` | 100–2000 | Linear | Number of trees (with early stopping at 50 rounds) |
| `max_depth` | 3–12 | Linear | Maximum tree depth — secondary regularizer |
| `learning_rate` | 0.005–0.3 | Log | Shrinkage per tree |
| `num_leaves` | 16–256 | Linear | **Primary capacity control** — max leaves per tree |
| `subsample` | 0.5–1.0 | Linear | Row sampling ratio (can interact with GOSS) |
| `colsample_bytree` | 0.5–1.0 | Linear | Feature sampling ratio per tree |
| `min_child_samples` | 5–100 | Linear | Minimum samples in a leaf — prevents overfitting on small partitions |
| `reg_alpha` | 1e-8–10.0 | Log | L1 regularization on leaf weights |
| `reg_lambda` | 1e-8–10.0 | Log | L2 regularization on leaf weights |

The key difference from XGBoost's search space: **`num_leaves`** is the primary knob. In XGBoost, `max_depth` controls capacity. In LightGBM, `num_leaves` does, because leaf-wise growth doesn't fill all levels.

---

## 7. LightGBM vs. XGBoost: Key Differences

| Aspect | XGBoost | LightGBM |
|--------|---------|----------|
| **Split finding** | Pre-sorted (exact) or histogram | Histogram-only (faster) |
| **Tree growth** | Level-wise | Leaf-wise (best-first) |
| **Primary capacity** | `max_depth` | `num_leaves` |
| **Speed** | Slower on large datasets | 10–20× faster |
| **Memory** | Higher (float storage) | Lower (1-byte bins) |
| **Data sampling** | Random subsample | GOSS (gradient-aware) |
| **Feature reduction** | None built-in | EFB (sparse bundling) |
| **Accuracy** | Strong baseline | Comparable or slightly better |
| **Overfitting risk** | Balanced trees, more predictable | Leaf-wise can overfit on small data |

---

## 8. Strengths and Weaknesses

### Strengths

- **Speed**: The histogram + leaf-wise combination makes LightGBM the fastest GBDT for large datasets
- **Memory efficiency**: 1-byte per feature + histogram subtraction trick minimizes memory footprint
- **Scalability**: Handles millions of rows and thousands of features efficiently
- **Native categorical support**: Can handle categoricals directly without one-hot encoding (via optimal split-finding on categories)
- **Distributed training**: Built-in support for multi-machine training

### Weaknesses

- **Overfitting on small datasets**: Leaf-wise growth is aggressive — needs careful tuning of `num_leaves` and `min_child_samples`
- **Sensitivity to `num_leaves`**: Too many leaves overfits; too few underfits. More sensitive than XGBoost's `max_depth`
- **Same fundamental limitations as all GBDTs**: Axis-aligned splits, no extrapolation, no representation learning

---

## 9. Key References

| Reference | Contribution |
|-----------|--------------|
| Friedman (2001) | Gradient Boosting Machine |
| Chen & Guestrin (2016) | XGBoost — the baseline LightGBM improves upon |
| Ke et al. (2017) | **LightGBM** — histogram splitting, leaf-wise growth, GOSS, EFB |
| Shi et al. (2007) | Histogram-based methods (precursor idea) |

---

## 10. In This Thesis

LightGBM represents the **speed-optimized GBDT**. Compared to XGBoost:
- It should achieve similar accuracy with faster training (especially on larger datasets like covertype, year_prediction)
- The `num_leaves` parameter makes it more flexible but also more prone to overfitting on small datasets (credit_g, phoneme)

**Preprocessing**: Same as XGBoost — ordinal-encode categoricals, median-impute missing values.
