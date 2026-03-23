# Analysis Framework — Research Roadmap

> **Purpose**: This document defines how the experimental results will be analyzed, segmented, and interpreted. It serves as the blueprint for the entire analytical phase — from raw experiment outputs to thesis-level conclusions. It also maps how this project feeds into the three-stage academic pipeline: Individual Development 1, Individual Development 2, and the Master's Thesis.

---

## Table of Contents

1. [Academic Pipeline Overview](#1-academic-pipeline-overview)
2. [What We Have (Infrastructure Inventory)](#2-what-we-have-infrastructure-inventory)
3. [Analysis Criteria (Segmentation Axes)](#3-analysis-criteria-segmentation-axes)
4. [Analysis Plan Per Criterion](#4-analysis-plan-per-criterion)
5. [Cross-Criterion Synthesis](#5-cross-criterion-synthesis)
6. [Pipeline Investigation: Are We Getting the Best from Each Model?](#6-pipeline-investigation)
7. [Paths Toward Individual Development 2 and the Thesis](#7-paths-toward-the-thesis)
8. [Execution Order and Dependencies](#8-execution-order-and-dependencies)
9. [Expected Deliverables](#9-expected-deliverables)

---

## 1. Academic Pipeline Overview

This project serves three academic milestones, each building on the previous:

### Individual Development 1 — The Comprehensive Benchmark

**Goal**: Run all experiments, produce results, and deliver a multi-criteria comparative analysis of 11 models across 18 datasets.

**Deliverable**: A complete empirical study with:
- Per-criterion segmented analysis (see Section 4)
- Statistical significance tests and critical difference diagrams
- Clear, evidence-backed answers to Research Questions 1–4
- Identified failure modes and open questions for each model family

**This is the foundation.** Everything else depends on these results being complete, correct, and deeply analyzed.

### Individual Development 2 — Refinement or Advancement

**Goal**: Take a specific finding from the benchmark and develop it into a focused contribution. Examples:

- "Model X underperforms in scenario Y because of limitation Z — here is a modification that addresses Z"
- "The preprocessing pipeline for family F is suboptimal for datasets with property P — here is an improved pipeline"
- "None of the DL models handle extreme class imbalance well — here is an augmentation or loss modification that helps"

**This requires the benchmark to reveal concrete, actionable insights.** The analysis framework below is designed to surface these.

### Master's Thesis — The Full Arc

**Goal**: Benchmark → Insight → Contribution. The thesis tells a complete story:

1. We surveyed and benchmarked the state-of-the-art (Individual Development 1)
2. We identified a specific gap or limitation (transition from 1 to 2)
3. We proposed and validated a solution (Individual Development 2)
4. We placed our contribution in the broader context of the field

The contribution could be:
- A novel method for tabular data
- A principled modification to an existing architecture
- A hybrid approach that combines strengths of different families
- A new preprocessing/embedding strategy
- A practical decision framework backed by rigorous evidence

**The benchmark must be designed to reveal where that contribution lives.**

---

## 2. What We Have (Infrastructure Inventory)

### 2.1 Models (11)

| ID | Model | Family | Key Property | Year |
|----|-------|--------|-------------|------|
| 1 | MLP | DL | Baseline neural network | — |
| 2 | XGBoost | GBDT | Industry standard, histogram-based | 2016 |
| 3 | LightGBM | GBDT | Speed-optimized, leaf-wise growth | 2017 |
| 4 | CatBoost | GBDT | Native categorical handling, ordered boosting | 2018 |
| 5 | TabNet | DL | Sequential attention for feature selection | 2021 |
| 6 | FT-Transformer | DL | Feature tokenization + self-attention | 2021 |
| 7 | SAINT | DL | Dual attention (feature + inter-sample) | 2021 |
| 8 | TabM | DL | Parameter-efficient batch ensembling | 2024/2025 |
| 9 | RealMLP | DL | Strong defaults, smart preprocessing | 2024 |
| 10 | STab | DL | Stochastic competition (LWTA + Embedding Mixture) | 2024 |
| 11 | TabPFN v2 | FM | Zero-shot foundation model, pre-trained on synthetic data | 2025 |

**Family distribution**: 3 GBDT, 7 DL, 1 Foundation Model

**Design rationale**: The DL family is intentionally overrepresented because it contains the most diversity — attention-based (FT-Transformer, SAINT, STab), MLP-based (MLP, RealMLP, TabM), and attention+feature-selection (TabNet). Each brings a different inductive bias.

### 2.2 Datasets (18)

| Dataset | Task | Samples | Features | Types | Imbalance | Domain |
|---------|------|---------|----------|-------|-----------|--------|
| credit_g | Binary | ~1,000 | 20 | Mixed | Moderate | Finance |
| phoneme | Binary | ~5,400 | 5 | Numerical | Moderate | Signal |
| bank_customer_churn | Binary | ~10,000 | 13 | Mixed | 4:1 | Banking |
| magictelescope | Binary | ~19,000 | 10 | Numerical | Balanced | Physics |
| amazon_employee | Binary | ~33,000 | 9 | Categorical | High | HR |
| bank_marketing | Binary | ~45,000 | 16 | Mixed | High | Banking |
| adult | Binary | ~49,000 | 14 | Mixed | 3:1 | Census |
| cardiovascular_disease | Binary | ~70,000 | 11 | Mixed | Balanced | Healthcare |
| higgs | Binary | ~98,000 | 28 | Numerical | Balanced | Physics |
| give_me_some_credit | Binary | ~150,000 | 10 | Numerical | 14:1 | Finance |
| helena | Multiclass (100) | ~65,000 | 27 | Numerical | — | Mixed |
| jannis | Multiclass (4) | ~84,000 | 54 | Numerical | — | Mixed |
| covertype | Multiclass (7) | ~581,000 | 54 | Numerical | — | Forestry |
| wine_quality | Regression | ~6,500 | 11 | Numerical | — | Chemistry |
| california_housing | Regression | ~21,000 | 8 | Numerical | — | Real estate |
| superconduct | Regression | ~21,000 | 81 | Numerical | — | Materials |
| diamonds | Regression | ~54,000 | 9 | Mixed | — | Commerce |
| year_prediction | Regression | ~515,000 | 90 | Numerical | — | Music |

**Size distribution**:
- Small (< 10K): credit_g, phoneme, bank_customer_churn, wine_quality — 4 datasets
- Medium (10K–50K): magictelescope, amazon_employee, bank_marketing, adult, california_housing, superconduct — 6 datasets
- Large (50K–100K): cardiovascular_disease, higgs, helena, jannis, diamonds — 5 datasets
- Very large (> 100K): give_me_some_credit, covertype, year_prediction — 3 datasets

**Feature type distribution**: Numerical-only: 11, Mixed: 6, Categorical-only: 1

**Task distribution**: Binary: 10, Multiclass: 3, Regression: 5

### 2.3 Metrics

| Task | Primary | Secondary |
|------|---------|-----------|
| Binary | ROC-AUC | Accuracy, F1, Log-loss, KS |
| Multiclass | Log-loss | Accuracy, F1-macro, F1-weighted |
| Regression | RMSE | MAE, R² |

**Temporal metrics** (collected per experiment):
- Hyperparameter optimization time (Optuna wall-clock)
- Training time (final model fit)
- Inference time (predict on test set)

### 2.4 Statistical Tests

- Friedman test (omnibus: "is there any difference among models?")
- Nemenyi post-hoc test (pairwise: "which models differ?")
- Pairwise Wilcoxon signed-rank with Holm-Bonferroni correction
- Cohen's d effect sizes (practical significance, not just statistical)
- Bootstrap 95% confidence intervals
- Critical difference diagrams

### 2.5 Analysis Notebooks (existing)

| Notebook | Current Focus | Status |
|----------|-------------|--------|
| 01 — Data Exploration | Dataset statistics, distributions, imbalance | Ready (runs on data alone) |
| 02 — Results Overview | Heatmaps, ranks, win/loss, CV variance | Awaiting results |
| 03 — Statistical Tests | Friedman, Nemenyi, Wilcoxon, CD diagrams | Awaiting results |
| 04 — Scaling Analysis | Performance vs dataset size, vs features | Awaiting results |
| 05 — Cost Analysis | Training time, tuning cost, Pareto front | Awaiting results |
| 06 — Interpretability | Feature importance, SHAP, model agreement | Awaiting results |

---

## 3. Analysis Criteria (Segmentation Axes)

Based on the professor's guidance, the analysis must be segmented across multiple orthogonal criteria. Each criterion answers a different practical question.

| # | Criterion | Question It Answers |
|---|-----------|-------------------|
| C1 | **Predictive Performance** | Which model family is most accurate, and is the difference statistically significant? |
| C2 | **Dataset Size Sensitivity** | How does performance change with dataset scale? Where do DL/FM models become competitive? |
| C3 | **Feature Dimensionality** | Do high-dimensional datasets favor different models than low-dimensional ones? |
| C4 | **Feature Type Sensitivity** | How do models handle numerical-only vs mixed vs categorical-only data? |
| C5 | **Class Imbalance Robustness** | Which models degrade under severe imbalance, and by how much? |
| C6 | **Computational Cost** | What is the total compute budget (tuning + training + inference) per model? |
| C7 | **Inference Latency** | Which models are viable for real-time or batch production? |
| C8 | **Tuning Sensitivity** | How much does performance depend on hyperparameter optimization? What if you use defaults? |
| C9 | **Interpretability** | Can practitioners understand why a model made a prediction? |
| C10 | **Practical Robustness** | How stable are rankings across CV folds? How often does a model catastrophically fail? |

---

## 4. Analysis Plan Per Criterion

### C1: Predictive Performance (Global)

**Data source**: `results/aggregated/test_results.csv`

**Analyses**:
1. **Per-task heatmap**: models × datasets, colored by primary metric. Shows the full picture at a glance.
2. **Aggregated ranks**: Average rank per model across all datasets of each task type. The core summary statistic.
3. **Statistical tests**: Friedman → Nemenyi/Wilcoxon → CD diagrams. Per task type.
4. **Effect sizes**: Cohen's d between all model pairs. Distinguishes "statistically significant but practically irrelevant" from "meaningfully better."
5. **Win/loss/tie matrix**: For each model pair, count how many datasets model A beats model B (and by how much).
6. **Family-level aggregation**: Average performance of GBDT family vs DL family vs FM. Are the families distinguishable, or is within-family variance larger than between-family variance?

**Key plots**: Performance heatmap, CD diagram, rank distribution box plot per family.

**Notebook**: 02 (heatmaps, ranks, win/loss) + 03 (statistical tests).

**Expected insight**: Whether the "GBDTs always win on tabular data" narrative holds when using 2024–2025 DL architectures with proper tuning.

---

### C2: Dataset Size Sensitivity

**Data source**: `test_results.csv` merged with dataset metadata (approx_samples).

**Analyses**:
1. **Size-group comparison**: Split datasets into Small (< 10K), Medium (10K–50K), Large (50K–100K), Very Large (> 100K). Compute average rank per model within each group.
2. **Performance trend**: Scatter plot of primary metric vs. log(n_samples), one series per model. Add LOESS or linear trend lines.
3. **Crossover analysis**: Identify the sample size threshold (if any) where the best DL model starts matching or exceeding the best GBDT. This is Research Question 2.
4. **TabPFN degradation curve**: Specific analysis of how TabPFN v2 performance drops as dataset size increases beyond its pre-training regime (~10K). Compare against Lucca's findings.
5. **Per-group statistical tests**: Run Friedman + Nemenyi within each size group separately. Rankings may differ by group.

**Key plots**: Performance vs. n_samples scatter with trend lines; group-wise CD diagrams; bar chart of average rank per size group.

**Notebook**: 04 (scaling analysis).

**Expected insight**: The precise dataset size regime where DL models become competitive. Whether TabM/RealMLP/STab behave differently from FT-Transformer/SAINT in large datasets.

---

### C3: Feature Dimensionality

**Data source**: `test_results.csv` merged with dataset metadata (approx_features).

**Analyses**:
1. **Dimensionality-group comparison**: Split into Low (≤ 10 features), Medium (11–30), High (> 30). Average rank per model within each group.
2. **Performance vs. n_features scatter**: Like C2 but on the feature axis.
3. **Interaction with dataset size**: 2D grid (size × features) to detect interaction effects. Some models may excel with "many samples + few features" but struggle with "few samples + many features."
4. **High-dimensional stress test**: Focus on superconduct (81 features), year_prediction (90 features), jannis (54), covertype (54). Do attention-based models learn better feature interactions in high dimensions?

**Key plots**: Performance vs. n_features scatter; size × features interaction heatmap.

**Notebook**: 04 (scaling analysis, second half).

**Expected insight**: Whether attention-based models (FT-Transformer, SAINT, STab) have an advantage in high-dimensional datasets where pairwise feature interactions matter more.

---

### C4: Feature Type Sensitivity

**Data source**: `test_results.csv` merged with dataset metadata (feature_types).

**Analyses**:
1. **Feature type group comparison**: Split datasets into Numerical-only (11), Mixed (6), Categorical-only (1). Average rank per model within each group.
2. **CatBoost advantage**: CatBoost has native categorical handling. Does it outperform other GBDTs specifically on mixed/categorical datasets? Quantify the delta.
3. **DL preprocessing impact**: DL models use one-hot encoding for categoricals (expanding dimensionality). Does this hurt performance on datasets like amazon_employee (all categorical)?
4. **Encoding strategy investigation**: If a DL model underperforms on categorical data, is it the model or the encoding? (This could seed an Individual Development 2 project.)

**Key plots**: Grouped bar chart of performance by feature type; CatBoost rank delta on categorical vs. numerical datasets.

**Notebook**: New analysis section in 04 or a dedicated notebook.

**Expected insight**: Whether DL models' reliance on one-hot encoding is a bottleneck on categorical-heavy datasets. This directly connects to potential pipeline improvements (Section 6).

---

### C5: Class Imbalance Robustness

**Data source**: `test_results.csv` for binary classification datasets only, merged with imbalance ratio metadata.

**Analyses**:
1. **Imbalance profile**: Compute the actual positive-class ratio for each binary dataset. Sort datasets by imbalance severity.
2. **Performance vs. imbalance ratio**: Plot primary metric (ROC-AUC) and KS against imbalance ratio. One series per model.
3. **Degradation rate**: For each model, fit a linear model of performance ~ log(imbalance_ratio). The slope measures sensitivity to imbalance. Steeper negative slopes = more sensitive.
4. **KS as complementary metric**: KS directly measures class separability and may reveal degradation patterns that ROC-AUC masks (since ROC-AUC is threshold-invariant but KS is not).
5. **give_me_some_credit deep dive**: This dataset has 14:1 imbalance and 150K samples. Which models handle this realistic, extreme-imbalance scenario best?

**Key plots**: Performance vs. imbalance ratio scatter; degradation slope comparison bar chart; KS vs. ROC-AUC correlation per model.

**Notebook**: New dedicated analysis, potentially 07_imbalance_analysis.ipynb.

**Expected insight**: Whether any model family is systematically fragile under imbalance. Lucca found TabPFN v2 is particularly sensitive — we'll verify this at larger scale. This could directly motivate a cost-sensitive or augmentation-based contribution for Individual Development 2.

---

### C6: Computational Cost

**Data source**: `test_results.csv` (tuning_time_s) + `fold_results.csv` (train_time_s).

**Analyses**:
1. **Total compute budget**: For each model, compute `tuning_time + training_time` summed across all datasets. This is the real-world cost of using a model in a benchmark setting.
2. **Per-dataset cost breakdown**: Stacked bar chart: tuning | training | inference. Per model, per dataset.
3. **Performance vs. compute Pareto front**: Scatter (x = total compute time, y = average performance). Draw the Pareto front. Models on the front are efficient; models below it are dominated.
4. **Cost-normalized performance**: Define `efficiency = performance / log(compute_time)`. Rank models by this metric. This directly answers Research Question 3.
5. **Family-level cost comparison**: Average compute per family (GBDT, DL, FM). The cost ratio between families is a key practical number.
6. **TabPFN cost paradox**: TabPFN has zero tuning cost but potentially high inference cost. Compute its total lifecycle cost (including inference on all test sets) vs. the others.

**Key plots**: Pareto front scatter; compute budget bar chart; efficiency ranking.

**Notebook**: 05 (cost analysis).

**Expected insight**: The concrete answer to "how much more does a DL model cost than XGBoost, and is the performance gain worth it?"

---

### C7: Inference Latency

**Data source**: `test_results.csv` or `fold_results.csv` (inference time per prediction batch).

**Analyses**:
1. **Absolute latency comparison**: Box plot of inference times per model across all datasets.
2. **Latency scaling**: Inference time vs. dataset size (n_samples × n_features). Do some models scale worse?
3. **Production viability classification**: Define thresholds:
   - Real-time (< 10ms per batch): viable for API serving
   - Batch (< 1s): viable for batch pipelines
   - Slow (< 60s): viable for offline analysis only
   - Impractical (> 60s): not viable for most production use
4. **STab Bayesian averaging cost**: STab runs N=64 forward passes at inference. Quantify the latency multiplier compared to a single-pass model.
5. **TabPFN inference anomaly**: Lucca found TabPFN inference takes 128s average, peaking at 904s. Verify and explain (likely due to in-context learning complexity).

**Key plots**: Latency box plot; latency vs. dataset size scatter; production viability classification chart.

**Notebook**: 05 (cost analysis, inference section).

**Expected insight**: Clear, actionable guidance on which models can be deployed in production. This is one of the most practically valuable outputs of the thesis.

---

### C8: Tuning Sensitivity

**Data source**: Optuna study objects (stored in result JSONs) + comparison against default hyperparameters.

**Analyses**:
1. **Tuning gain**: For each model, compare best-tuned performance vs. default-hyperparameter performance. The delta measures tuning sensitivity.
2. **Convergence speed**: How many Optuna trials does each model need before converging to near-optimal performance? Plot metric vs. trial number.
3. **TabPFN advantage**: TabPFN requires zero tuning. If its performance is within X% of tuned GBDTs, the zero-tuning property is extremely valuable in practice.
4. **Hyperparameter stability**: Across datasets, do optimal hyperparameters cluster in similar regions, or does each dataset need very different settings? High variance = high tuning sensitivity.
5. **Recommended defaults**: From the tuning results, compute the median or mode of optimal hyperparameters across all datasets. These become "recommended defaults" — a practical contribution.

**Key plots**: Tuning gain bar chart; Optuna convergence curves; hyperparameter distribution violin plots.

**Notebook**: Extension to 05, or standalone 08_tuning_analysis.ipynb.

**Expected insight**: Whether the additional compute cost of hyperparameter tuning is justified for each model. If a model works well with defaults, tuning is wasted compute.

---

### C9: Interpretability

**Data source**: Trained models (GBDT feature importance, SHAP values).

**Analyses**:
1. **Feature importance comparison**: For datasets where "ground truth" feature relevance is known or intuitive (e.g., adult: education and occupation should matter), compare what each model considers important.
2. **SHAP analysis**: TreeExplainer for GBDTs (exact), DeepExplainer or KernelExplainer for DL models. Compare SHAP summary plots across families.
3. **Model agreement**: Cohen's kappa between all model pairs' predictions. High agreement + same performance → the model choice doesn't matter. Low agreement + different performance → the model captures different patterns.
4. **Interpretability taxonomy**: Classify each model family:
   - GBDT: Intrinsically interpretable (feature importance, split values, SHAP)
   - DL (attention-based): Partially interpretable (attention weights, but debated reliability)
   - DL (MLP-based): Black box (requires post-hoc methods like SHAP)
   - FM (TabPFN): Opaque (pre-trained, no direct feature attribution)

**Key plots**: SHAP beeswarm plots; inter-model agreement heatmap; feature importance rank correlation matrix.

**Notebook**: 06 (interpretability).

**Expected insight**: Whether interpretability is a meaningful differentiator. In regulated industries (finance, healthcare), interpretability may override raw performance.

---

### C10: Practical Robustness

**Data source**: `fold_results.csv` (per-fold performance across outer CV).

**Analyses**:
1. **Rank stability**: For each model, compute the standard deviation of its rank across CV folds. Low std = stable model.
2. **Catastrophic failure rate**: Count how many (model, dataset) pairs have a metric value more than 2σ below the model's average. Some DL models may occasionally diverge or fail to train.
3. **Worst-case analysis**: For each model, report its worst performance across all datasets. A model that averages well but occasionally catastrophically fails is risky.
4. **Training convergence**: For DL models, track how often early stopping triggers before max_epochs. Frequent early stopping at few epochs may indicate training instability.

**Key plots**: Rank stability bar chart; failure rate matrix; worst-case performance comparison.

**Notebook**: Extension to 02 (CV variance section) or standalone.

**Expected insight**: Whether any DL model is too unreliable for practical use, even if its average performance is competitive.

---

## 5. Cross-Criterion Synthesis

After completing the per-criterion analyses, the synthesis produces the thesis-level conclusions.

### 5.1 The Decision Matrix

Create a single summary table that scores each model across all 10 criteria:

| Model | C1 Perf | C2 Scale | C3 Dims | C4 Types | C5 Imbal | C6 Cost | C7 Latency | C8 Tuning | C9 Interp | C10 Robust |
|-------|---------|----------|---------|----------|----------|---------|------------|-----------|-----------|------------|
| XGBoost | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? |
| ... | | | | | | | | | | |

Each cell gets a qualitative score (★★★, ★★, ★) derived from the quantitative analysis. This becomes the practitioner's reference card.

### 5.2 The Practical Decision Flowchart

Derived from the decision matrix, create a flowchart:

```
Is dataset size > 100K?
  ├── Yes → Is inference latency critical?
  │         ├── Yes → XGBoost / LightGBM
  │         └── No  → Is best performance critical?
  │                   ├── Yes → [Best-performing model for large datasets]
  │                   └── No  → XGBoost (safe default)
  └── No  → Is dataset size < 5K?
            ├── Yes → TabPFN v2 (zero-shot, no tuning needed)
            └── No  → Is compute budget limited?
                      ├── Yes → RealMLP (strong defaults, minimal tuning)
                      └── No  → [Best-performing model for medium datasets]
```

This flowchart is a **concrete, actionable contribution** of the thesis. It doesn't exist in the literature for this model set.

### 5.3 Family-Level Narrative

Synthesize the per-criterion findings into a narrative for each family:

- **GBDT family**: Strengths, weaknesses, when to use, when not to use.
- **DL family**: Which architectures are worth the complexity overhead and when.
- **Foundation model (TabPFN)**: Where zero-shot makes sense, where it doesn't.

---

## 6. Pipeline Investigation: Are We Getting the Best from Each Model? <a id="6-pipeline-investigation"></a>

The professor specifically flagged this: *"Pode ser que algum algoritmo esteja fazendo um pipeline que não esteja melhorando 100% o aspecto de uso daquele pipeline."*

This is critical. If a model underperforms, is it the model's fault or our pipeline's fault?

### 6.1 Preprocessing Audit

| Family | Current Pipeline | Potential Issues |
|--------|-----------------|------------------|
| GBDT | Ordinal-encode cats, median-impute | **CatBoost**: Ordinal encoding discards its native categorical handling. Should we pass raw categoricals to CatBoost instead? |
| DL | StandardScaler + one-hot cats + median-impute | **One-hot explosion**: amazon_employee has 9 categorical features with potentially high cardinality. One-hot encoding could create hundreds of sparse features, hurting attention-based models. Target encoding or entity embeddings might be better. |
| FM | Same as GBDT (ordinal + impute) | **TabPFN**: v2.5 handles missing values natively. Are we imputing unnecessarily? |

### 6.2 Per-Model Fairness Check

For each model, verify:
1. **Is the search space reasonable?** Compare our ranges against the original paper's recommendations and Lucca's tuned values.
2. **Is early stopping calibrated?** 20 epochs patience with 200 max may be too aggressive for some models (STab) or too lenient for others (MLP).
3. **Are we using the model's best features?** E.g., TabNet has built-in feature selection — are we leveraging it? STab has Bayesian averaging — is N=64 the right inference sample count?
4. **Is the batch size appropriate?** SAINT's inter-sample attention depends on batch composition. Batch size 256 may be too small for it to work well.

### 6.3 Investigation Protocol

If a model underperforms expectations:

1. **Check the Optuna convergence curve**: Did 100 trials suffice? If the curve is still improving at trial 100, the model is under-tuned.
2. **Check the training curve**: Is the model overfitting? Underfitting? Early stopping too early?
3. **Run a targeted ablation**: Try the paper's recommended defaults instead of Optuna's best. If paper defaults beat Optuna, our search space is wrong.
4. **Check the preprocessing**: Run the model with an alternative preprocessing (e.g., target encoding instead of one-hot for categoricals) and compare.

**These investigations are not just debugging — they are potential contributions.** If we discover that "STab with target encoding outperforms STab with one-hot encoding by 3 points on categorical datasets," that's a publishable finding.

---

## 7. Paths Toward Individual Development 2 and the Thesis <a id="7-paths-toward-the-thesis"></a>

The benchmark results will reveal opportunities. Here are the most likely paths, ordered by feasibility:

### Path A: Practical Decision Framework (Low Risk, High Impact)

**Contribution**: The decision matrix and flowchart from Section 5.2, backed by the most comprehensive and rigorous benchmark to date (11 models, 18 datasets, nested CV, statistical tests).

**Why it's valuable**: No published paper provides this for the 2024–2025 model set (TabM, RealMLP, STab, TabPFN v2). Practitioners need this.

**Thesis title example**: *"When to Use What: A Rigorous Multi-Criteria Guide for Tabular Machine Learning in 2025"*

### Path B: Improved Pipeline for an Underperforming Model (Medium Risk, High Novelty)

**Contribution**: Identify a model that underperforms due to a pipeline issue (preprocessing, training schedule, loss function) rather than an architectural limitation. Propose and validate a fix.

**Concrete examples**:
- **Cost-sensitive STab**: If STab degrades under imbalance, modify the KL-regularized loss to incorporate class weights. The variational framework makes this natural.
- **Categorical embedding for DL models**: Replace one-hot encoding with learned entity embeddings (like CatBoost's native handling). Test whether this closes the gap between DL and GBDT on categorical-heavy datasets.
- **Adaptive Bayesian averaging for STab**: Instead of fixed N=64, use adaptive N based on prediction entropy. Confident predictions use fewer samples.

**Why it's valuable**: Shows deep understanding of the model internals and produces a concrete, reproducible improvement.

### Path C: Hybrid Approach (Higher Risk, Highest Novelty)

**Contribution**: Combine strengths of different families into a new approach.

**Concrete examples**:
- **GBDT feature selection + Transformer prediction**: Use XGBoost's feature importance to pre-select the top-K features, then feed only those into a Transformer. Tests whether reducing noise helps attention-based models.
- **Stacking / meta-learning**: Train a meta-learner that selects the best model per dataset based on meta-features (size, dimensionality, imbalance ratio, feature type ratio).
- **STab + TabPFN embedding**: Use TabPFN's frozen encoder to generate embeddings, then feed them into STab's hybrid architecture instead of raw features.

**Why it's valuable**: Novel contribution that goes beyond benchmarking. Higher risk but higher thesis-level impact.

### Path D: Foundation Model Investigation (Medium Risk, Timely)

**Contribution**: Deep-dive into why TabPFN v2 fails on large datasets or imbalanced data, and propose mitigations.

**Concrete examples**:
- **Subsampling strategies**: Test different subsampling approaches (random, stratified, clustering-based) for datasets that exceed TabPFN's capacity.
- **Fine-tuning TabPFN**: Explore whether fine-tuning TabPFN's frozen encoder on domain-specific data improves performance (the v2 paper suggests this is possible but doesn't benchmark it extensively).
- **Calibration analysis**: TabPFN's pre-training on synthetic data may produce miscalibrated probabilities on real data. Measure and correct.

**Why it's valuable**: Foundation models for tabular data are the hottest topic in the field right now (Nature 2025 publication). Any contribution here is highly citable.

**Note**: The path will become clear once the benchmark results are in. Don't choose now — let the data speak.

---

## 8. Execution Order and Dependencies

### Phase 1: Run Experiments (Priority: CRITICAL)

```
Step 1: Download all 18 datasets
        python scripts/download_data.py

Step 2: Run full pipeline (11 models × 18 datasets = 198 experiments)
        python scripts/run_all.py --gpu 0
        
        Estimated time: 1-2 weeks on a single T4 GPU
        - GBDTs: ~2-3 hours total
        - DL models (FT-T, SAINT, STab, TabNet, TabM, RealMLP, MLP): ~5-10 days
        - TabPFN: ~1-2 hours total
        
        The pipeline is resumable. If interrupted, re-run the same command.

Step 3: Aggregate results
        python scripts/evaluate.py
```

### Phase 2: Per-Criterion Analysis (Priority: HIGH)

```
Step 4: Run notebook 01 (data exploration) — can run immediately, no experiments needed
Step 5: Run notebook 02 (results overview) — requires Step 3
Step 6: Run notebook 03 (statistical tests) — requires Step 3
Step 7: Run notebook 04 (scaling analysis) — requires Step 3
Step 8: Run notebook 05 (cost analysis) — requires Step 3
Step 9: Create notebook 07 (imbalance analysis) — requires Step 3, binary results only
Step 10: Run notebook 06 (interpretability) — requires trained models, can run in parallel with Steps 5-9
```

### Phase 3: Cross-Criterion Synthesis (Priority: HIGH)

```
Step 11: Build decision matrix (Section 5.1)
Step 12: Build practical flowchart (Section 5.2)
Step 13: Write family-level narratives (Section 5.3)
```

### Phase 4: Pipeline Investigation (Priority: MEDIUM)

```
Step 14: Preprocessing audit (Section 6.1) — analyze after Phase 2 reveals anomalies
Step 15: Targeted ablations for underperforming models (Section 6.2)
```

### Phase 5: Individual Development 2 Direction (Priority: AFTER Phase 2-3)

```
Step 16: Based on findings, choose a path from Section 7
Step 17: Implement and validate
```

---

## 9. Expected Deliverables

### For Individual Development 1

1. **All 198 experiment result JSONs** (11 models × 18 datasets)
2. **Aggregated CSVs**: fold results, test results, CV summary
3. **Statistical test outputs**: Friedman, Nemenyi, Wilcoxon, Cohen's d (per task type)
4. **Critical difference diagrams** (per task type)
5. **Per-criterion analysis** (Sections C1–C10), documented in notebooks
6. **Decision matrix** (Section 5.1)
7. **Practical decision flowchart** (Section 5.2)
8. **Written report** covering all of the above

### For Individual Development 2

9. **Identified gap or limitation** (from Phase 4 investigation)
10. **Proposed solution** (from Section 7 paths)
11. **Validation experiments** (ablation studies, comparison against benchmark baseline)
12. **Written report** covering the contribution

### For the Master's Thesis

13. **All of the above**, integrated into a coherent narrative
14. **Literature review** positioned against the current state-of-the-art
15. **Original contribution** (Path A, B, C, or D)
16. **Conclusion with future work** directions
