# Foundation Models vs Gradient-Boosted Decision Trees on Tabular Data

A rigorous empirical comparison of gradient-boosted decision trees, deep learning architectures, and foundation models on tabular prediction tasks. This repository contains the full experimental pipeline for a master's thesis that benchmarks 10 models across 15 OpenML datasets, using nested cross-validation with Bayesian hyperparameter optimization, followed by statistical significance testing and multi-dimensional analysis (scaling behavior, computational cost, and model interpretability).

## Research Questions

1. **Performance**: Do deep learning and foundation models match or surpass GBDTs on standard tabular benchmarks?
2. **Scaling**: At what dataset sizes or feature dimensionalities do DL/foundation models begin to outperform GBDTs?
3. **Cost**: What is the performance-vs-compute trade-off across model families?
4. **Behavior**: Do different model families learn different decision boundaries, and when does each family excel?

## Models

Ten models organized into three families, each receiving family-appropriate preprocessing:

| Family | Model | Key Reference |
|---|---|---|
| **GBDT** | XGBoost | Chen & Guestrin, 2016 |
| | LightGBM | Ke et al., 2017 |
| | CatBoost | Prokhorenkova et al., 2018 |
| **Deep Learning** | FT-Transformer | Gorishniy et al., 2021 |
| | TabNet | Arik & Pfister, 2021 |
| | SAINT | Somepalli et al., 2021 |
| | TabM | Gorishniy et al., 2024 (ICLR 2025) |
| | RealMLP | Holzmuller et al., 2024 (NeurIPS 2024) |
| | MLP (baseline) | -- |
| **Foundation Model** | TabPFN v2 | Hollmann et al., 2025 |

**Preprocessing by family:**
- **GBDT**: ordinal-encode categoricals, median-impute missing values
- **Deep Learning**: standardize numericals, one-hot encode categoricals, median-impute
- **Foundation Model (TabPFN)**: minimal preprocessing (ordinal-encode categoricals); TabPFN handles normalization internally

## Datasets

Fifteen datasets sourced from OpenML, spanning binary classification, multiclass classification, and regression:

| Dataset | Task | Samples | Features | Feature Types |
|---|---|---|---|---|
| adult | Binary | ~49,000 | 14 | Mixed |
| bank_marketing | Binary | ~45,000 | 16 | Mixed |
| amazon_employee | Binary | ~33,000 | 9 | Categorical |
| higgs | Binary | ~98,000 | 28 | Numerical |
| magictelescope | Binary | ~19,000 | 10 | Numerical |
| phoneme | Binary | ~5,400 | 5 | Numerical |
| credit_g | Binary | ~1,000 | 20 | Mixed |
| covertype | Multiclass (7) | ~581,000 | 54 | Numerical |
| jannis | Multiclass (4) | ~84,000 | 54 | Numerical |
| helena | Multiclass (100) | ~65,000 | 27 | Numerical |
| california_housing | Regression | ~21,000 | 8 | Numerical |
| wine_quality | Regression | ~6,500 | 11 | Numerical |
| diamonds | Regression | ~54,000 | 9 | Mixed |
| superconduct | Regression | ~21,000 | 81 | Numerical |
| year_prediction | Regression | ~515,000 | 90 | Numerical |

## Experimental Protocol

```
For each (model, dataset) pair:
  1. Hold out 20% of data as the final test set (stratified for classification)
  2. Hyperparameter tuning on the remaining 80% pool:
     - 3-fold inner cross-validation
     - 100 Optuna TPE trials with MedianPruner
     - TabPFN is zero-shot (no tuning)
  3. Outer evaluation with best hyperparameters:
     - 5-fold cross-validation on the pool (for variance estimation)
  4. Final evaluation:
     - Retrain on ~90% of pool, evaluate on held-out test set
```

**Metrics:**
- Binary classification: ROC-AUC (primary), accuracy, F1, log-loss
- Multiclass classification: log-loss (primary), accuracy, F1-macro, F1-weighted
- Regression: RMSE (primary), MAE, R^2

**Statistical tests:** Friedman test, Nemenyi post-hoc, pairwise Wilcoxon signed-rank (Holm-Bonferroni corrected), Cohen's d effect sizes, bootstrap 95% confidence intervals, critical difference diagrams.

**Reproducibility:** Global seed = 42 (Python, NumPy, PyTorch, CUDA determinism). Runtime environment captured automatically.

## Repository Structure

```
.
├── configs/
│   ├── datasets.yaml          # Dataset registry (OpenML IDs, task types, metadata)
│   ├── models.yaml             # Hyperparameter search spaces per model
│   └── experiment.yaml         # Global experiment settings (folds, trials, seeds)
├── data/
│   ├── raw/                    # Downloaded datasets (parquet, gitignored)
│   └── processed/              # Preprocessed cache (gitignored)
├── notebooks/
│   ├── 01_data_exploration.ipynb    # Dataset statistics and distributions
│   ├── 02_results_overview.ipynb    # Performance heatmaps and summary tables
│   ├── 03_statistical_tests.ipynb   # Friedman, Nemenyi, Wilcoxon, CD diagrams
│   ├── 04_scaling_analysis.ipynb    # Performance vs dataset size and features
│   ├── 05_cost_analysis.ipynb       # Training time and compute trade-offs
│   └── 06_interpretability.ipynb    # Feature importance and model agreement
├── results/
│   ├── raw/                    # Per-experiment JSON results (gitignored)
│   ├── aggregated/             # Aggregated CSVs (gitignored)
│   ├── figures/                # Generated plots (gitignored)
│   └── logs/                   # Run logs (gitignored)
├── scripts/
│   ├── download_data.py        # Download all datasets from OpenML
│   ├── train.py                # Train/tune a single model+dataset
│   ├── run_all.py              # Orchestrate the full pipeline
│   └── evaluate.py             # Aggregate results and run statistical tests
├── src/
│   ├── data/
│   │   ├── download.py         # OpenML download logic
│   │   ├── registry.py         # Dataset loading, splitting, CV fold generation
│   │   └── preprocessing.py    # Per-family preprocessing pipelines
│   ├── models/
│   │   ├── base.py             # Abstract model interface
│   │   ├── factory.py          # Model registry and factory
│   │   ├── xgboost_model.py
│   │   ├── lightgbm_model.py
│   │   ├── catboost_model.py
│   │   ├── ft_transformer.py
│   │   ├── tabnet_model.py
│   │   ├── saint_model.py      # Vendored PyTorch implementation
│   │   ├── tabm_model.py       # Official tabm package (BatchEnsemble)
│   │   ├── realmlp_model.py   # pytabkit RealMLP-TD wrapper
│   │   ├── tabpfn_model.py
│   │   └── mlp_model.py
│   ├── evaluation/
│   │   ├── metrics.py          # Classification and regression metrics
│   │   └── statistical_tests.py # Friedman, Nemenyi, Wilcoxon, CD diagrams
│   ├── tuning/
│   │   ├── search_spaces.py    # Optuna parameter suggestion per model
│   │   └── tuner.py            # Unified Optuna tuning loop with inner CV
│   └── utils/
│       ├── config.py           # YAML config loader (OmegaConf)
│       ├── environment.py      # Runtime environment capture
│       ├── logging.py          # Logging setup
│       ├── reproducibility.py  # Seed-setting for Python, NumPy, PyTorch
│       └── timer.py            # Context-manager timer
├── tests/                      # pytest test suite
├── pyproject.toml              # Project metadata and dependencies
└── .gitignore
```

## Setup and Installation

**Requirements:** Python >= 3.10, [uv](https://docs.astral.sh/uv/) (recommended) or pip.

```bash
# Clone the repository
git clone <repository-url>
cd Masters

# Install dependencies
uv sync

# Download all 15 datasets from OpenML
python scripts/download_data.py

# (Optional) Download a single dataset
python scripts/download_data.py --dataset wine_quality
```

## Running Experiments

### Single model on a single dataset

```bash
python scripts/train.py --model xgboost --dataset wine_quality
python scripts/train.py --model ft_transformer --dataset adult --gpu 0
```

### Full experimental pipeline

```bash
# Run all 10 models x 15 datasets (resumes from where it left off)
python scripts/run_all.py --gpu 0

# Run a subset
python scripts/run_all.py --models xgboost lightgbm --datasets adult wine_quality

# Force re-run (ignore existing results)
python scripts/run_all.py --gpu 0 --force
```

### Aggregate results and statistical tests

```bash
python scripts/evaluate.py
```

This reads all JSON files from `results/raw/`, produces aggregated CSVs in `results/aggregated/`, runs all statistical tests, and generates critical difference diagrams in `results/figures/`.

### Analysis notebooks

After running experiments, open the Jupyter notebooks for visualization and analysis:

```bash
jupyter notebook notebooks/
```

## Reproducing Results

All experiments are deterministic given the same seed, software versions, and hardware:

1. **Seed**: Set globally to 42 via `configs/experiment.yaml`. Covers Python's `random`, NumPy, PyTorch, and CUDA.
2. **Configs**: All experimental settings (fold counts, trial counts, metrics, paths) are declared in `configs/experiment.yaml`. Hyperparameter search spaces are in `configs/models.yaml`.
3. **Environment**: Running `scripts/run_all.py` automatically saves a snapshot of the runtime environment (Python version, package versions, GPU info) to `results/environment.json`.
4. **Dependencies**: Pinned in `pyproject.toml`. Use `uv sync` for exact dependency resolution.
5. **Resume**: `run_all.py` skips model-dataset pairs whose result JSON already exists, allowing safe interruption and resumption.

## Key Dependencies

| Package | Version | Purpose |
|---|---|---|
| Python | >= 3.10 | Runtime |
| numpy | >= 1.26 | Numerical computing |
| pandas | >= 2.1 | Data manipulation |
| scikit-learn | >= 1.4 | Preprocessing, metrics, CV |
| xgboost | >= 2.0 | XGBoost model |
| lightgbm | >= 4.2 | LightGBM model |
| catboost | >= 1.2 | CatBoost model |
| torch | >= 2.2 | PyTorch backend for DL models |
| rtdl_revisiting_models | >= 0.0.2 | FT-Transformer implementation |
| pytorch-tabnet | >= 4.1 | TabNet implementation |
| tabm | >= 0.0.3 | TabM (BatchEnsemble) implementation |
| pytabkit | >= 1.7 | RealMLP-TD implementation |
| tabpfn | >= 6.0 | TabPFN v2 foundation model |
| optuna | >= 3.5 | Bayesian hyperparameter optimization |
| scipy | >= 1.12 | Statistical tests |
| statsmodels | >= 0.14 | Multiple-testing correction |
| scikit-posthocs | >= 0.9 | Nemenyi post-hoc test |
| matplotlib | >= 3.8 | Visualization |
| seaborn | >= 0.13 | Statistical visualization |
| openml | >= 0.14 | Dataset downloads |

## License

*(To be added)*

## Citation

If you use this code or reference this work, please cite:

```bibtex
@mastersthesis{vasconcelos2026tabular,
  title  = {Foundation Models vs Gradient-Boosted Decision Trees on Tabular Data},
  author = {Vasconcelos, Gabriel},
  year   = {2026},
  school = {(University Name)},
  type   = {Master's Thesis}
}
```
