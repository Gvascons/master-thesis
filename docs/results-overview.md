> **NOTA (18/07/2026):** este documento retrata a fase de **11 modelos** do benchmark. O estado atual (14 modelos, 250/252 células, TabFM líder nas três tarefas) está em `notebooks/00_presentation.ipynb` (§6.2), `notebooks/TABELA_RESULTADOS.md` e `docs/programa-de-pesquisa.md`. Mantido como registro histórico da fase.

# Results Overview — Phase 1 (preliminary)

> **Status**: First pass over the complete benchmark (197/198 experiments;
> `tabpfn × helena` is N/A — 100 classes exceed TabPFN's 10-class limit).
> All numbers below are computed from `results/aggregated/test_results.csv`
> (hold-out test metrics, not the CV-fold metrics). Generated 2026-06-25.
>
> **Read this first — statistical power is low.** With 10 binary, 5 regression
> and 3 multiclass datasets, the critical difference is large (≈4.8 ranks for
> binary, ≈6.8 for regression). The Friedman test detects *that* a difference
> exists, but the post-hoc (Nemenyi) shows the **top cluster is statistically
> indistinguishable** — the only robust separations are at the bottom. Treat
> "winner" and average-rank orderings as descriptive, not as proof of
> superiority. This low-power caveat is itself a known property of small
> tabular benchmarks (Grinsztajn et al. 2022; McElfresh et al. 2024).

---

## 1. Headline

- **No statistically significant winner among the top models.** On binary,
  TabPFN, CatBoost, XGBoost, LightGBM, TabM and RealMLP are all mutually
  indistinguishable (Nemenyi p ≈ 0.5–1.0). Same on regression for the top
  cluster.
- **The one robust signal is at the bottom: TabNet is significantly the
  weakest** on both binary (vs TabPFN, p=0.001) and regression (vs TabPFN,
  p=0.025; vs MLP, p=0.035). MLP and STab are also consistently weak on
  binary/regression.
- **TabPFN is the standout on average**: best average rank on binary (3.2) and
  regression (2.0), and most per-dataset wins (8/18) — all **zero-shot, no
  tuning**. But it is not *significantly* ahead of the GBDTs, and its
  large-dataset results are on 50K stratified subsamples (its native limit).
- **The family ordering flips by task.** GBDTs/TabPFN lead binary and
  regression; **DL models lead multiclass** (STab, TabM, RealMLP top the 3
  multiclass datasets) — though with only 3 datasets this is descriptive.

---

## 2. Binary classification (10 datasets, primary metric ROC-AUC)

Friedman: χ² = 31.31, **p = 0.0005** (a difference exists somewhere).

| Model | Avg rank ↓ | Mean ROC-AUC | 95% bootstrap CI |
|-------|-----------:|-------------:|------------------|
| tabpfn | **3.2** | 0.8688 | [0.829, 0.908] |
| xgboost | 4.0 | 0.8666 | [0.828, 0.904] |
| catboost | 4.3 | **0.8717** | [0.834, 0.908] |
| lightgbm | 5.2 | 0.8658 | [0.828, 0.903] |
| tabm | 5.5 | 0.8607 | [0.824, 0.898] |
| realmlp | 6.2 | 0.8593 | [0.816, 0.901] |
| saint | 6.3 | 0.8552 | [0.815, 0.896] |
| ft_transformer | 6.4 | 0.8569 | [0.819, 0.895] |
| stab | 7.4 | 0.8426 | [0.792, 0.889] |
| mlp | 7.9 | 0.8560 | [0.820, 0.892] |
| tabnet | 9.6 | 0.8468 | [0.805, 0.889] |

**Careful note — best by *rank* ≠ best by *mean*.** TabPFN has the best average
rank (most consistent placement) but CatBoost has the highest mean ROC-AUC
(0.8717 vs 0.8688). The gap is ~0.003 AUC and all CIs overlap heavily, so
neither lead is meaningful. Nemenyi: only **tabpfn vs tabnet** is significant
(p=0.001); 3 of 55 pairs total.

## 3. Regression (5 datasets, primary metric RMSE)

Friedman: χ² = 23.56, **p = 0.0088**.

| Model | Avg rank ↓ |
|-------|-----------:|
| tabpfn | **2.0** |
| xgboost | 4.0 |
| realmlp | 4.2 |
| lightgbm | 4.6 |
| catboost | 4.8 |
| tabm | 6.4 |
| saint | 6.6 |
| ft_transformer | 7.2 |
| stab | 8.0 |
| mlp | 9.0 |
| tabnet | 9.2 |

Nemenyi: TabPFN is significantly better than **tabnet** (p=0.025) and **mlp**
(p=0.035) only; indistinguishable from the GBDTs and other DL.

## 4. Multiclass (3 datasets, primary metric log-loss) — descriptive only

**Caveat 1**: TabPFN is missing on helena (100-class limit), so the automated
pipeline drops helena and is left with 2 datasets — too few for any test. The
table below instead **excludes TabPFN** to keep all 3 datasets across the other
10 models. **Caveat 2**: with only 3 datasets this is descriptive, not
inferential (Friedman p=0.0265 is not trustworthy at N=3).

| Model | Avg rank ↓ (3 datasets, log-loss) |
|-------|-----------:|
| stab | 2.3 |
| tabm | 2.7 |
| realmlp | 3.0 |
| xgboost | 4.7 |
| ft_transformer | 5.0 |
| catboost | 5.7 |
| mlp | 6.3 |
| lightgbm | 6.7 |
| saint | 8.7 |
| tabnet | 10.0 |

On the 2 multiclass datasets it can handle, TabPFN posts the best log-loss on
covertype (0.164, the overall winner there) and a mid-pack 0.658 on jannis.

**This is the most interesting reversal in the benchmark**: STab, TabM and
RealMLP — mid/low on binary and regression — top the multiclass datasets, all
of which are high-class-count numerical problems (covertype 7, jannis 4,
helena 100). Worth a focused look in Phase 2 / ID2, but it needs more
multiclass datasets before any claim.

## 5. Per-dataset winners (by each dataset's primary metric)

| Dataset | Task | Winner | Value | Family |
|---------|------|--------|------:|--------|
| credit_g | binary | ft_transformer | 0.7873 | DL |
| phoneme | binary | tabpfn | 0.9529 | FM |
| telco_customer_churn | binary | tabpfn | 0.8490 | FM |
| magictelescope | binary | tabpfn | 0.9452 | FM |
| amazon_employee | binary | catboost | 0.8869 | GBDT |
| bank_marketing | binary | tabpfn | 0.9397 | FM |
| adult | binary | xgboost | 0.9301 | GBDT |
| cardiovascular_disease | binary | xgboost | 0.8002 | GBDT |
| higgs | binary | tabm | 0.8193 | DL |
| give_me_some_credit | binary | catboost | 0.8646 | GBDT |
| covertype | multiclass | tabpfn | 0.1644 | FM |
| jannis | multiclass | tabm | 0.6572 | DL |
| helena | multiclass | tabm | 2.5097 | DL |
| wine_quality | regression | xgboost | 0.5975 | GBDT |
| california_housing | regression | tabpfn | 0.2570 | FM |
| superconduct | regression | tabpfn | 0.8284 | FM |
| diamonds | regression | realmlp | 0.0630 | DL |
| year_prediction | regression | tabpfn | 8.5952 | FM |

**Wins by family**: FM (TabPFN, 1 model) **8** · GBDT (3 models) **5** · DL
(7 models) **5**. Per *model*, TabPFN's 8 wins from a single zero-shot network
is the most striking number — but again, most of these wins are inside the
statistical tie (point-estimate wins, CIs overlap).

## 6. Honest caveats (carry into the writeup)

1. **Low statistical power** (10/5/3 datasets). Friedman finds a difference;
   the top cluster is a tie. Don't claim a single best model.
2. **Rank vs mean disagree** on the binary #1 (TabPFN by rank, CatBoost by
   mean) — neither significant.
3. **TabPFN large-data results are on 50K subsamples**; its zero-shot,
   zero-tuning competitiveness is the real story, not raw dominance.
4. **Multiclass is under-powered and TabPFN-incomplete** (helena 100-class
   limit). The DL-leads-multiclass pattern is a hypothesis, not a result.
5. **GBDTs remain the pragmatic default**: top-cluster accuracy at a tiny
   fraction of the compute (see the cost analysis — Phase 2).

## 7. What this sets up

- The performance picture alone does not separate the field → the thesis
  contribution must come from the **other axes** (compute, latency, dataset
  size, imbalance, interpretability) where the families *do* separate clearly,
  and from the **task-dependent reversals** (DL on multiclass; attention models
  failing on high-cardinality categoricals).
- Concrete Phase-2 next steps: cost-vs-performance Pareto (notebook 05),
  size-sensitivity (notebook 04), imbalance robustness (KS metric), and a
  proper multiclass run that either adds datasets or formally excludes TabPFN.
