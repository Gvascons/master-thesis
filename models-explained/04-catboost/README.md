# CatBoost (Categorical Boosting)

> **Paper**: Prokhorenkova et al., "CatBoost: unbiased boosting with categorical features", NeurIPS 2018  
> **Role in this thesis**: The GBDT with native categorical handling and ordered boosting. CatBoost addresses two subtle but important issues in standard gradient boosting: target leakage from categorical encoding and prediction shift from sequential tree building.

---

## 1. What It Is

CatBoost is a gradient-boosted decision tree framework from Yandex. While it shares the same additive ensemble structure as XGBoost and LightGBM, CatBoost introduces two fundamental innovations:

1. **Ordered Target Statistics** for categorical features — avoids target leakage
2. **Ordered Boosting** — eliminates the prediction shift inherent in standard gradient boosting

Additionally, CatBoost uses **oblivious decision trees** (symmetric trees) as its base learners, which have useful properties for speed and regularization.

---

## 2. Key Innovation #1: Ordered Target Statistics (Categorical Encoding)

### The Problem: Target Leakage

The standard approach for encoding categorical features in GBDTs is target encoding — replacing each category with the mean of the target variable for that category:

```
Category "A" → mean(y where x = "A") = 0.73
Category "B" → mean(y where x = "B") = 0.21
```

The problem: this uses **all** training examples to compute the encoding, including the example being predicted. This creates **target leakage** — information from the label leaks into the feature, leading to overfitting. The model memorizes categories that appear few times.

### CatBoost's Solution: Ordered Target Statistics

CatBoost computes target statistics using only **preceding** examples in a random permutation:

```
Random permutation: σ = [x₃, x₁, x₇, x₂, x₅, ...]

For sample x₇ (3rd in permutation):
  TS(x₇) = [Σ yⱼ for j ∈ {x₃, x₁} where category matches + prior] / (count + 1)
```

Each sample's encoding is computed using **only samples that come before it** in the random ordering. This eliminates target leakage because no sample sees its own label or future labels.

Multiple random permutations are used during training to reduce variance from any single ordering.

### Why This Matters

On datasets with high-cardinality categoricals (like `amazon_employee` in this thesis, which has 9 categorical features), target leakage is a real problem. Standard target encoding overfits on rare categories. CatBoost's ordered statistics provide unbiased estimates, leading to better generalization — particularly when categories have few observations.

---

## 3. Key Innovation #2: Ordered Boosting

### The Problem: Prediction Shift

In standard gradient boosting, each tree is fit to the residuals (gradients) of the current ensemble. But these residuals are computed using the **same training data** that was used to build the previous trees. This creates a subtle bias called **prediction shift**:

```
Standard boosting:
  Tree 1 trained on all data
  Residuals computed on the same data     ← biased! Tree 1 already "saw" this data
  Tree 2 trained on these biased residuals
  Residuals computed on the same data     ← still biased
  ...
```

The gradients computed on the training data are systematically different from what they would be on unseen data. This is a form of overfitting that gets worse with more trees.

### CatBoost's Solution: Ordered Boosting

CatBoost maintains multiple models trained on different prefixes of a random permutation:

```
Permutation σ = [x₃, x₁, x₇, x₂, x₅, ...]

Model M₁ trained on {x₃}
Model M₂ trained on {x₃, x₁}
Model M₃ trained on {x₃, x₁, x₇}
...

For computing the gradient for x₇:
  Use M₂ (trained WITHOUT x₇) to predict x₇
  → unbiased gradient estimate
```

Each sample's gradient is computed using a model that was **never trained on that sample**. This eliminates prediction shift.

In practice, maintaining N separate models is expensive. CatBoost approximates this with a small number of permutations and a clever bookkeeping scheme using the same oblivious tree structure.

---

## 4. Oblivious Decision Trees (Symmetric Trees)

CatBoost uses **oblivious (symmetric) decision trees** as base learners. Every node at the same depth uses the **same split condition**:

```
Standard tree:              Oblivious tree:
    [age < 45?]                [age < 45?]
   /           \              /            \
[income < 50?] [edu > 12?]  [income < 50?] [income < 50?]   ← same split!
 /   \          /   \         /   \          /   \
□     □        □     □       □     □        □     □
```

### Properties

| Property | Effect |
|----------|--------|
| **Regularity** | All nodes at the same depth share the same feature and threshold. Fewer parameters per tree. |
| **Speed** | The symmetric structure enables SIMD (vectorized) evaluation — the same comparison is applied to all samples simultaneously. |
| **Regularization** | Fewer unique split conditions = less model complexity = resistance to overfitting. |
| **Lookup table** | A depth-`d` oblivious tree has exactly `2^d` leaves. Prediction is a bitmask lookup: each split contributes one bit to the leaf index. |

The tradeoff: oblivious trees are less expressive than unconstrained trees for the same depth. CatBoost compensates by using more iterations.

---

## 5. Hyperparameters (Search Space in This Thesis)

| Parameter | Range | Scale | Effect |
|-----------|-------|-------|--------|
| `iterations` | 100–2000 | Linear | Number of trees (with early stopping at 50 rounds) |
| `depth` | 3–10 | Linear | Tree depth (oblivious trees, so exactly 2^depth leaves) |
| `learning_rate` | 0.005–0.3 | Log | Shrinkage per tree |
| `l2_leaf_reg` | 1.0–10.0 | Linear | L2 regularization on leaf weights (λ in the objective) |
| `bagging_temperature` | 0.0–1.0 | Linear | Controls the intensity of Bayesian bootstrap sampling. 0 = uniform weights (no bagging), 1 = Exponential(1) weights (standard Bayesian bootstrap). Higher values → more aggressive resampling. |
| `random_strength` | 0.0–10.0 | Linear | Score randomization for split selection. Adds noise to split scores to reduce overfitting. |

Notable differences from XGBoost/LightGBM:
- **`bagging_temperature`**: CatBoost uses Bayesian bootstrap rather than uniform subsampling. The temperature controls the weight distribution.
- **`random_strength`**: Unique to CatBoost — adds random noise to split scores during tree building, acting as a regularizer.

---

## 6. CatBoost vs. XGBoost vs. LightGBM

| Aspect | XGBoost | LightGBM | CatBoost |
|--------|---------|----------|----------|
| **Tree type** | Unconstrained | Unconstrained | Oblivious (symmetric) |
| **Categorical handling** | Ordinal encoding | Native or ordinal | **Ordered target statistics** |
| **Boosting** | Standard | Standard | **Ordered boosting** |
| **Split finding** | Exact or histogram | Histogram | Histogram (symmetric) |
| **Target leakage** | Possible | Possible | **Addressed by design** |
| **Prediction shift** | Present | Present | **Addressed by design** |
| **Speed** | Moderate | Fastest | Moderate (inference fast due to oblivious trees) |
| **Default performance** | Needs tuning | Needs tuning | Strong out-of-box |

### When CatBoost Excels

- **High-cardinality categoricals**: Its core advantage. Datasets like `amazon_employee`, `adult`, and `bank_marketing` with many categorical features.
- **Small datasets**: Ordered boosting and ordered target statistics provide better regularization on limited data.
- **Out-of-the-box**: CatBoost's defaults are carefully chosen — less tuning needed than XGBoost/LightGBM.

### When It Doesn't

- **All-numerical datasets**: No categorical advantage. The oblivious tree constraint may be a slight disadvantage.
- **Very large datasets**: Ordered boosting with permutations adds overhead. LightGBM is faster.
- **Deep interactions**: Oblivious trees need more depth than unconstrained trees to capture the same interactions.

---

## 7. Strengths and Weaknesses

### Strengths

- **Principled categorical handling**: No manual feature engineering for categoricals — the algorithm handles it optimally
- **Statistically principled**: Both ordered target statistics and ordered boosting have theoretical guarantees against bias
- **Fast inference**: Oblivious trees enable vectorized, branchless prediction
- **Robust defaults**: Strong out-of-box performance with minimal tuning
- **GPU training**: Efficient GPU implementation (especially for oblivious trees)

### Weaknesses

- **Training speed**: Ordered boosting with multiple permutations is slower than standard boosting
- **Memory**: Multiple permutations and model copies require more memory
- **Less flexible trees**: Oblivious trees are more constrained than unconstrained trees
- **Same fundamental GBDT limitations**: Axis-aligned splits, no extrapolation, no representation learning

---

## 8. Key References

| Reference | Contribution |
|-----------|--------------|
| Micci-Barreca (2001) | Target encoding for categoricals (precursor, with leakage) |
| Chen & Guestrin (2016) | XGBoost |
| Ke et al. (2017) | LightGBM |
| Prokhorenkova et al. (2018) | **CatBoost** — ordered target statistics, ordered boosting, oblivious trees |
| Hancock & Khoshgoftaar (2020) | Survey of categorical encoding methods |

---

## 9. In This Thesis

CatBoost completes the **GBDT triumvirate** (XGBoost, LightGBM, CatBoost). Its distinctive value is:
- On datasets with categoricals (`adult`, `bank_marketing`, `amazon_employee`, `credit_g`, `diamonds`), it should shine due to ordered target statistics
- On pure numerical datasets, it should perform similarly to XGBoost and LightGBM
- Its ordered boosting provides a theoretical advantage on small datasets like `credit_g` and `phoneme`

**Preprocessing**: Ordinal-encode categoricals (CatBoost receives the `cat_feature_indices` to handle them internally), median-impute missing values.

**Implementation note**: The wrapper passes `cat_feature_indices` to CatBoost's `fit()` so it can apply ordered target statistics on the correct features. XGBoost and LightGBM don't receive this parameter.
