# Multi-Layer Perceptron (MLP)

> **Role in this thesis**: Baseline deep learning model. Every other neural architecture is measured against the MLP to justify its added complexity.

---

## 1. What It Is

The Multi-Layer Perceptron is the foundational feedforward neural network. It consists of fully-connected (dense) layers stacked sequentially, where every neuron in one layer connects to every neuron in the next. Despite its simplicity, the MLP is a **universal function approximator** — given sufficient width and depth, it can approximate any continuous function to arbitrary precision (Cybenko, 1989; Hornik et al., 1989).

In the context of tabular data, the MLP remains a surprisingly strong baseline. Gorishniy et al. (2021) showed that a well-tuned MLP often matches or exceeds more complex deep learning architectures on tabular benchmarks, a finding later reinforced by the TabArena benchmark (Holzmüller et al., 2024).

---

## 2. Architecture

### 2.1 Building Block

Each layer in our implementation follows the pattern:

```
Linear → BatchNorm → ReLU → Dropout
```

| Component | Purpose |
|-----------|---------|
| **Linear** (`nn.Linear`) | Learnable affine transformation: `y = Wx + b`. Each neuron computes a weighted sum of all inputs plus a bias. |
| **Batch Normalization** (`nn.BatchNorm1d`) | Normalizes activations to zero mean, unit variance across the mini-batch. Stabilizes training, enables higher learning rates, and provides mild regularization (Ioffe & Szegedy, 2015). |
| **ReLU** (`nn.ReLU`) | Activation function: `f(x) = max(0, x)`. Introduces non-linearity — without it, stacking linear layers would collapse into a single linear transformation. ReLU is preferred for its computational simplicity and resistance to vanishing gradients. |
| **Dropout** (`nn.Dropout`) | During training, randomly zeroes a fraction `p` of activations. Forces the network to learn redundant representations, reducing overfitting (Srivastava et al., 2014). At inference time, all neurons are active (with outputs scaled by `1-p`). |

### 2.2 Full Network

```
Input (n_features)
  → [Linear → BatchNorm → ReLU → Dropout] × n_blocks
  → Linear (d_hidden → d_out)
  → Output
```

The final linear layer (the "head") maps the last hidden representation to the output space:
- **Binary classification**: 1 output → sigmoid → probability
- **Multiclass classification**: `n_classes` outputs → softmax → class probabilities  
- **Regression**: 1 output → raw value

### 2.3 Mathematical Formulation

For a network with `L` hidden layers, the forward pass computes:

```
h₀ = x                                    (input features)
zₗ = Wₗ · hₗ₋₁ + bₗ                      (linear transformation)
z̃ₗ = BatchNorm(zₗ)                        (normalization)
aₗ = ReLU(z̃ₗ) = max(0, z̃ₗ)              (activation)
hₗ = Dropout(aₗ, p)                       (regularization)
ŷ  = W_head · h_L + b_head                (output projection)
```

Where `Wₗ ∈ ℝ^(dₗ × dₗ₋₁)` and `bₗ ∈ ℝ^dₗ` are learnable parameters.

---

## 3. Training Procedure

### 3.1 Loss Functions

| Task | Loss | Formula |
|------|------|---------|
| Binary | `BCEWithLogitsLoss` | `-[y·log(σ(ŷ)) + (1-y)·log(1-σ(ŷ))]` |
| Multiclass | `CrossEntropyLoss` | `-Σᵢ yᵢ·log(softmax(ŷ)ᵢ)` |
| Regression | `MSELoss` | `(y - ŷ)²` |

`BCEWithLogitsLoss` combines sigmoid + binary cross-entropy in a single numerically stable operation (log-sum-exp trick), which is why the model outputs raw logits rather than probabilities.

### 3.2 Optimizer

**AdamW** (Loshchilov & Hutter, 2019) — Adam with decoupled weight decay. Standard Adam applies L2 regularization to the gradient, which interferes with the adaptive learning rate. AdamW applies weight decay directly to the parameters, making regularization strength independent of the learning rate.

### 3.3 Early Stopping

The model trains for up to `max_epochs` (default: 200), monitoring validation loss after each epoch. If validation loss doesn't improve for `patience` (default: 20) consecutive epochs, training halts and the best checkpoint is restored. This prevents overfitting — the model returns to the point of lowest validation loss.

### 3.4 Mini-Batch Training

Data is processed in batches (default: 256 samples). This provides:
- **Computational efficiency**: GPU parallelism over the batch dimension
- **Implicit regularization**: Gradient noise from sampling batches acts as a regularizer
- **Memory management**: Full dataset doesn't need to fit in GPU memory at once

---

## 4. Hyperparameters (Search Space in This Thesis)

| Parameter | Range | Scale | Effect |
|-----------|-------|-------|--------|
| `n_blocks` | 1–4 | Linear | Depth. More blocks = more representational capacity, but also harder to train and more prone to overfitting. |
| `d_hidden` | {64, 128, 256, 512} | Categorical | Width. Wider layers capture more complex feature interactions per layer. |
| `dropout` | 0.0–0.5 | Linear | Regularization strength. Higher = more aggressive dropping. Too high starves the network of capacity. |
| `learning_rate` | 1e-5 – 1e-3 | Log | Step size for AdamW. Too large → divergence. Too small → slow convergence or getting stuck. |
| `weight_decay` | 1e-6 – 1e-3 | Log | L2 penalty on weights. Prevents weights from growing unboundedly large. |

The interplay between `dropout` and `weight_decay` is key — both are regularizers, and over-regularizing (high values for both) can underfit.

---

## 5. Why It Matters for Tabular Data

### Strengths

- **Simplicity**: No inductive bias about data structure, which can be an advantage when the true structure is unknown
- **Universal approximation**: Can theoretically model any function
- **Fast inference**: Single forward pass, no tree traversals or attention computations
- **GPU-friendly**: Matrix multiplications are perfectly suited for GPU parallelism
- **Strong baseline**: When properly tuned, surprisingly competitive (Gorishniy et al., 2021; Kadra et al., 2021)

### Weaknesses

- **No feature interaction modeling by design**: Must learn all interactions from scratch through gradient descent. GBDTs get axis-aligned splits for free; transformers get pairwise attention. The MLP must discover which features interact and how, using only dense matrix multiplications.
- **Sensitive to hyperparameters**: Performance varies dramatically with depth, width, learning rate, and regularization. A poorly tuned MLP is much worse than a poorly tuned GBDT.
- **Data-hungry**: Needs more samples to learn the same relationships that GBDTs capture with fewer observations, because it lacks the inductive bias of tree-based splits.
- **No native handling of categoricals**: Requires preprocessing (one-hot or entity embeddings). GBDTs handle categoricals natively through splits.
- **No built-in feature selection**: Processes all features equally. Irrelevant features add noise to the gradient signal.

### The "Regularization Cocktail" Insight

Kadra et al. (2021) showed that a standard MLP with a carefully chosen combination of regularization techniques (dropout, weight decay, batch normalization, data augmentation, early stopping) can match the performance of far more complex architectures on tabular data. This underscores that for tabular problems, **the training procedure often matters more than the architecture**.

---

## 6. Key References

| Reference | Contribution |
|-----------|--------------|
| Rosenblatt (1958) | The Perceptron — single-layer predecessor |
| Rumelhart, Hinton & Williams (1986) | Backpropagation — made training MLPs practical |
| Cybenko (1989) | Universal approximation theorem for single hidden layer |
| Hornik et al. (1989) | Extended universal approximation to arbitrary depth |
| Srivastava et al. (2014) | Dropout regularization |
| Ioffe & Szegedy (2015) | Batch Normalization |
| Loshchilov & Hutter (2019) | AdamW optimizer |
| Gorishniy et al. (2021) | "Revisiting Deep Learning Models for Tabular Data" — showed tuned MLPs are competitive |
| Kadra et al. (2021) | "Well-tuned Simple Nets Excel on Tabular Datasets" — regularization cocktail |

---

## 7. In This Thesis

The MLP serves as the **deep learning floor**. If a more complex architecture (FT-Transformer, SAINT, TabM, etc.) doesn't significantly outperform the MLP on a dataset, that complexity isn't justified. Conversely, datasets where the MLP struggles but GBDTs excel reveal structural properties (e.g., many categoricals, small sample size) that favor tree-based approaches.

**Preprocessing**: Standardize numericals (zero mean, unit variance), one-hot encode categoricals, median-impute missing values — the standard deep learning pipeline.
