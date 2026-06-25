# Cost vs Performance — Phase 1 (preliminary)

> Companion to `results-overview.md`. Where raw performance is a statistical
> tie (top cluster indistinguishable), **cost is the criterion that actually
> separates the model families.** All times are wall-clock on the study machine
> (RTX 5080 GPU for DL/TabPFN, CPU for GBDTs); they reflect realistic
> deployment cost on this hardware, not a controlled FLOPs comparison.
> Generated 2026-06-25 from the raw result JSONs.

## What is (and isn't) measured

- **Final-model training time** (`test_metrics.train_time_s`): time to fit one
  model with fixed hyperparameters. **Clean** — independent of how many tuning
  trials were run, so it is the fair cost axis used below.
- **Total tuning time** (`tuning_time_s`): HPO wall-clock. Reflects the actual
  protocol — **GBDT = 100 trials, DL = 25 trials, TabPFN = 0** — so DL's
  per-trial cost is ~4× higher than the totals imply. The `adult` dataset is
  **excluded** from tuning aggregates because its DL runs used 100 trials
  (Phase-1 validation anchor) vs 25 elsewhere.
- **Inference latency: NOT captured** in this run (only fit time was recorded).
  This is a gap to fill with a dedicated measurement pass before the inference
  criterion in the framework can be reported.

## 1. Final-model training time (median over 18 datasets)

| Model | Family | Median (s) | Mean (s) | Max (s) |
|-------|--------|-----------:|---------:|--------:|
| lightgbm | GBDT | **0.66** | 5.2 | 58 |
| xgboost | GBDT | **0.69** | 15.6 | 163 |
| tabpfn | FM | 1.35 | 6.2 | 48 |
| catboost | GBDT | 3.76 | 38.0 | 387 |
| mlp | DL | 12.95 | 26.9 | 97 |
| tabm | DL | 33.76 | 62.9 | 167 |
| realmlp | DL | 36.90 | 45.2 | 100 |
| tabnet | DL | 64.97 | 131.7 | 569 |
| ft_transformer | DL | 87.22 | 124.1 | 662 |
| stab | DL | 198.85 | 484.5 | 2942 |
| saint | DL | 202.19 | 387.3 | 1618 |

The spread is ~300× (LightGBM 0.66 s → SAINT/STab ~200 s median). GBDTs and
TabPFN train in ~1–4 s; the heaviest DL models take minutes per fit.

## 2. Total tuning cost (excl. adult; real protocol)

| Model | Family | Median/exp (s) | Total (h) |
|-------|--------|---------------:|----------:|
| tabpfn | FM | **0** | **0.00** |
| lightgbm | GBDT | 104 | 5.5 |
| xgboost | GBDT | 157 | 11.4 |
| mlp | DL | 1,117 | 8.2 |
| realmlp | DL | 1,352 | 8.9 |
| tabm | DL | 2,242 | 20.0 |
| catboost | GBDT | 646 | 34.9 |
| tabnet | DL | 5,928 | 38.0 |
| ft_transformer | DL | 4,505 | 43.4 |
| saint | DL | 9,831 | 84.6 |
| stab | DL | 14,622 | 119.7 |

TabPFN's zero tuning cost is a categorical advantage: it reaches top-cluster
rank with **no search at all**. Among the rest, the heaviest DL models spent
40–120 h of GPU search — and recall this is at only 25 trials; at the GBDTs'
100 trials it would be ~4× that.

## 3. Cost–performance Pareto frontier (training time vs average rank)

A model is **Pareto-optimal** if no other model is both cheaper to train *and*
better-ranked. Ranks are each task's primary metric (multiclass excludes
TabPFN to keep all 3 datasets — see overview caveat).

### Binary (10 datasets) — frontier: **TabPFN, XGBoost, LightGBM**
| Model | rank | train (s) | Pareto |
|-------|-----:|----------:|--------|
| tabpfn | 3.2 | 1.35 | ★ |
| xgboost | 4.0 | 0.69 | ★ |
| catboost | 4.3 | 3.76 | dominated (by xgboost) |
| lightgbm | 5.2 | 0.66 | ★ |
| tabm | 5.5 | 33.76 | dominated |
| realmlp … tabnet | 6.2–9.6 | 13–202 | all dominated |

### Regression (5 datasets) — frontier: **TabPFN, XGBoost, LightGBM**
| Model | rank | train (s) | Pareto |
|-------|-----:|----------:|--------|
| tabpfn | 2.0 | 1.35 | ★ |
| xgboost | 4.0 | 0.69 | ★ |
| realmlp | 4.2 | 36.90 | dominated |
| lightgbm | 4.6 | 0.66 | ★ |
| catboost | 4.8 | 3.76 | dominated |
| (all DL) | 6.4–9.2 | 13–202 | dominated |

### Multiclass (3 datasets, descriptive) — frontier: **STab, TabM, XGBoost, LightGBM**
| Model | rank | train (s) | Pareto |
|-------|-----:|----------:|--------|
| stab | 2.3 | 198.85 | ★ |
| tabm | 2.7 | 33.76 | ★ |
| realmlp | 3.0 | 36.90 | dominated (by tabm) |
| xgboost | 4.7 | 0.69 | ★ |
| lightgbm | 6.7 | 0.66 | ★ |
| (rest) | 5.0–10.0 | — | dominated |

## 4. Reading

1. **On binary and regression, no deep-learning model is cost-efficient.**
   The entire Pareto frontier is {TabPFN, XGBoost, LightGBM}. Every DL model is
   strictly dominated — it costs 10–300× more to train for an equal-or-worse
   rank. CatBoost is also dominated (XGBoost is both faster and better-ranked).
2. **TabPFN is the efficiency standout**: best rank on both tasks, ~1 s to fit,
   and zero tuning. The earlier caveats hold (50K subsample on large data;
   no >10-class support).
3. **Multiclass is the one place DL earns its cost.** STab and TabM make the
   frontier because they genuinely rank best. But **TabM gives ~most of STab's
   quality (rank 2.7 vs 2.3) at 6× lower training cost** (34 s vs 199 s) — so
   even here the heaviest model is not the rational pick. N=3, so this is a
   hypothesis to test with more multiclass datasets.
4. **The thesis hinge**: because raw accuracy is a tie, the decision collapses
   onto cost — and on cost the answer is unambiguous for binary/regression
   (GBDT/TabPFN). The genuinely open question, and a natural **AI-2 direction**,
   is *why* DL reverses ahead on multiclass and whether that holds up.

## 5. Gaps to close in Phase 2

- **Inference latency** — not recorded; needs a dedicated timing pass (TabPFN
  is expected to flip from cheap-to-train to expensive-at-inference, since it
  carries the training set as context).
- **Compute-normalized performance** — e.g. AUC per GPU-second — once latency
  is in.
- **More multiclass datasets** to turn the DL-leads-multiclass observation from
  descriptive (N=3) into something testable.
