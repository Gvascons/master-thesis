# Segmented Analysis — Phase 1 (preliminary)

> Companion to `results-overview.md` (raw performance) and `cost-analysis.md`
> (training cost). Where the headline benchmark is a statistical tie in the top
> cluster, this document asks **where the families separate along three operational
> axes**: inference latency, dataset-size sensitivity, and class-imbalance
> robustness. All numbers were **recomputed from scratch** from
> `results/aggregated/test_results.csv` and the raw dataset loaders on
> 2026-06-25 — none are carried over from earlier notes.
>
> **Read the caveats.** N is tiny (10 binary / 5 regression / 3 multiclass), the
> "within-dataset rank" is across the 11 models, and the per-model Spearman
> trends below are mostly **not** statistically significant. Treat them as
> descriptive direction-of-effect, not proof. Confounds are stated inline and
> are real — especially that large datasets are capped at 100K samples and the
> three multiclass datasets are all medium/large.

---

## Axis 1 — Inference latency

**Why it matters.** Training cost (`cost-analysis.md`) already separates the
families, but it rewards TabPFN, which trains in ~1 s. Inference latency is the
*opposite* axis: a model trained once and served many times is dominated by
per-row predict cost. TabPFN is expected to flip from cheapest-to-train to
most-expensive-to-serve because it carries the training set as in-context
"memory" and re-attends to it on every prediction.

**Protocol.** Single dataset (`adult`, binary, 48,842 rows). Each model is
loaded with its tuned `best_params` from the saved experiment JSONs, fit on the
training pool, then `predict()` is timed over the 9,769-row hold-out test set,
median of 5 repeats after a warm-up pass, with CUDA synchronisation. STab uses
its full `n_inference_samples=64` Bayesian averaging (the deployment setting).
Reported as microseconds per row. Single dataset / single machine
(RTX 5080) — directional, not a controlled FLOPs comparison.

| Model | Family | µs / row | Total for 9,769 rows |
|-------|--------|---------:|---------------------:|
| xgboost | GBDT | **0.33** | 3.2 ms |
| catboost | GBDT | 1.43 | 13.9 ms |
| realmlp | DL | 1.52 | 14.8 ms |
| lightgbm | GBDT | 1.91 | 18.7 ms |
| mlp | DL | 2.37 | 23.2 ms |
| tabm | DL | 3.48 | 34.0 ms |
| tabnet | DL | 17.09 | 166.9 ms |
| ft_transformer | DL | 28.87 | 282.1 ms |
| saint | DL | 44.69 | 436.6 ms |
| tabpfn | FM | **7,416** | **72.4 s** |
| stab | DL | **8,645** | **84.5 s** |

The spread is **~26,000×** (XGBoost 0.33 µs/row → STab 8,645 µs/row). Three
tiers emerge: a sub-5 µs/row "instant" tier, a 17–45 µs/row attention tier, and
two models three-plus orders of magnitude slower.

### Reading

1. **Latency is *not* a trees-vs-DL split — it's an architecture split.** The
   fast tier (< 5 µs/row) is all three GBDTs **plus** the MLP-family deep nets
   (RealMLP 1.52, MLP 2.37, TabM 3.48). A plain MLP serves as fast as a boosted
   tree. What's slow is *mechanism*: attention (TabNet/FT-T/SAINT, 17–45 µs/row,
   1–2 orders up but still sub-half-second total), and the two models that do
   heavy per-row work — TabPFN (in-context re-attention over the training set)
   and STab (64 stochastic forward passes for Bayesian averaging).
2. **TabPFN's predicted reversal is confirmed and it is dramatic.** It is the
   cheapest model to *train* (~1 s, zero tuning — see `cost-analysis.md`) and
   the **second most expensive to *serve***: scoring the 9,769-row test set
   takes **72 seconds** vs 3 ms for XGBoost — about **22,000× slower**. Its whole
   cost profile is inverted relative to every other model: free to fit, costly
   to query.
3. **STab is dominated on *both* cost axes.** It is the single most expensive
   model to train (≈120 GPU-h of tuning in `cost-analysis.md`) **and** the
   slowest at inference (8,645 µs/row). Its only redeeming result is the
   multiclass rank lead — bought at the highest total compute in the study, which
   sharpens the earlier point that TabM gives ~the same multiclass quality far
   more cheaply.
4. **For a latency-bound deployment the rational set is tiny**: XGBoost /
   CatBoost / LightGBM / RealMLP / MLP / TabM — all under ~3.5 µs/row, all within
   top-cluster accuracy. TabPFN is viable only for batch / low-QPS scoring; STab
   only where its multiclass edge is worth ~100 ms/row.

> Caveat: single dataset (`adult`), single GPU, `predict()` (label) timing.
> Latency scales with feature count and, for TabPFN, with training-set size — so
> absolute numbers will move across datasets; the **tier structure** (instant /
> attention / heavy) is the portable finding.

---

## Axis 2 — Dataset-size sensitivity

**Question.** Does a model's *relative* standing (its rank among the 11 on a
dataset) shift as datasets get larger? Size bins: **small** < 20k, **medium**
20k–100k, **large** ≥ 100k samples. Rank 1 = best on that dataset; lower mean
rank = better.

> **Confound — read first.** The pipeline subsamples any dataset above 100k to
> 100k *before* the split (TabPFN further to 50k), so the three "large" datasets
> (`give_me_some_credit` 150k, `year_prediction` 515k, `covertype` 581k) were
> actually trained on ~100k rows. The size axis therefore measures **20k vs 100k
> regimes, not true large-scale.** Separately, all three **multiclass** datasets
> (helena 65k, jannis 84k, covertype 581k) fall in medium/large, so any
> "DL improves with size" signal is partly the known DL-leads-multiclass effect.

### Family mean within-dataset rank, by size bin (all 18 datasets)

| Family | small (<20k) | medium (20–100k) | large (≥100k) |
|--------|-------------:|-----------------:|--------------:|
| foundation_model (TabPFN) | 2.17 | 3.25 | 2.00 |
| gbdt | 5.83 | 4.48 | 3.67 |
| deep_learning | 6.62 | 6.92 | 7.57 |

- **GBDTs improve monotonically with size** (5.83 → 4.48 → 3.67): the clearest
  trend on this axis.
- **TabPFN is strong everywhere and does *not* monotonically decay** (2.17 →
  3.25 → 2.00). An earlier note claimed TabPFN degrades on large data; the
  recomputed ranks do **not** support that — even on the capped/subsampled large
  sets it posts the best family rank. (It *is* handicapped by the 50k subsample;
  the point is only that its *rank* does not fall.)
- **Deep learning drifts slightly *worse* with size** (6.62 → 7.57) in aggregate
  — opposite to the common "DL needs big data" intuition, but see the per-model
  split below, which is more nuanced.

### Per-model trend: Spearman(rank vs log n) over all 18 datasets

| Model | ρ (rank vs log n) | p | reading |
|-------|------------------:|--:|---------|
| **stab** | **−0.64** | **0.004** | strongly improves with size (only significant trend) |
| realmlp | −0.22 | 0.38 | weakly improves |
| lightgbm | −0.25 | 0.32 | weakly improves |
| xgboost | −0.06 | 0.82 | flat |
| catboost | +0.14 | 0.59 | flat |
| tabpfn | +0.15 | 0.56 | flat (17 datasets; helena N/A) |
| mlp | +0.18 | 0.48 | flat |
| tabm | +0.23 | 0.37 | weakly worse |
| ft_transformer | +0.36 | 0.14 | weakly worse |
| tabnet | +0.41 | 0.09 | worse with size |

(ρ < 0 ⇒ rank number falls as n grows ⇒ the model gets *relatively better*.)

**Takeaway.** Only **STab** shows a statistically reliable size effect
(improves markedly on larger data, consistent with its multiclass strength).
Everything else is a non-significant drift. The aggregate "DL worsens with size"
is driven by tabnet/ft_transformer, **not** a family-wide law — STab and RealMLP
move the other way.

---

## Axis 3 — Class-imbalance robustness (binary)

**Question.** Among the 10 binary datasets, does relative rank track the class
imbalance ratio (majority : minority)? Primary metric is **ROC-AUC**, which is
threshold-independent and already fairly robust to prior shift — so this asks
"does ranking quality hold up under imbalance," not calibration. (KS is also
recorded and would be the sharper imbalance lens for Phase 2.)

Binary datasets by imbalance ratio:

| Dataset | imbalance | n | bin |
|---------|----------:|--:|-----|
| magictelescope | 1.00:1 | 13,376 | mild |
| cardiovascular_disease | 1.00:1 | 70,000 | mild |
| higgs | 1.12:1 | 98,050 | mild |
| credit_g | 2.33:1 | 1,000 | moderate |
| phoneme | 2.41:1 | 5,404 | moderate |
| telco_customer_churn | 2.77:1 | 7,043 | moderate |
| adult | 3.18:1 | 48,842 | moderate |
| bank_marketing | 7.55:1 | 45,211 | severe |
| give_me_some_credit | 13.96:1 | 150,000 | severe |
| amazon_employee | 16.27:1 | 32,769 | severe |

Bins: **mild** < 2:1, **moderate** 2–5:1, **severe** ≥ 5:1.

> **Confound — read first.** The severe bin is contaminated: `amazon_employee`
> (16:1) is also the **high-cardinality** dataset where attention models OOM'd
> and needed the one-hot cap (see `preprocessing-decisions.md`), and
> `give_me_some_credit` (14:1) is one of the capped large sets. So "DL struggles
> on severe imbalance" is partly "DL struggled on *this particular* dataset for
> an unrelated reason." Only 3 severe datasets.

### Family mean within-dataset rank, by imbalance bin (binary)

| Family | mild (<2:1) | moderate (2–5:1) | severe (≥5:1) |
|--------|------------:|-----------------:|--------------:|
| foundation_model (TabPFN) | 4.67 | 2.25 | 3.00 |
| gbdt | 5.56 | 4.33 | 3.67 |
| deep_learning | 6.38 | 7.25 | 7.43 |

- **GBDTs improve as imbalance grows** (5.56 → 3.67) — they take over the top of
  the ranking exactly where the minority class is rarest.
- **Deep learning degrades** (6.38 → 7.43), but with the amazon/cardinality
  confound above this is not clean.
- **TabPFN is best in the mild/moderate range** and stays competitive when
  severe.

### Per-model trend: Spearman(rank vs imbalance ratio), binary (n=10)

| Model | ρ | p | reading |
|-------|--:|--:|---------|
| catboost | −0.55 | 0.10 | improves with imbalance (strongest, not sig.) |
| xgboost | −0.39 | 0.27 | improves |
| lightgbm | −0.38 | 0.28 | improves |
| saint | +0.58 | 0.08 | degrades with imbalance (strongest DL) |
| mlp | +0.33 | 0.35 | degrades |
| tabm | +0.33 | 0.35 | degrades |
| others | |−0.1..+0.2| | flat |

**Takeaway.** Direction is consistent and intuitive — **GBDTs handle imbalance
best, attention/MLP DL worst, TabPFN in between** — but **no per-model
correlation reaches p < 0.05** at N=10, and the severe bin is confounded. This
is a hypothesis with the right sign, not an established result.

---

## Synthesis

| Axis | Clear separation? | Who wins | Strength of evidence |
|------|-------------------|----------|----------------------|
| Inference latency | **yes (decisive)** | fast tier: GBDTs + MLP-family DL (<3.5 µs/row) | single dataset, but ~26,000× spread |
| Dataset size | partial | GBDTs improve; STab improves (sig.); TabPFN flat & strong | 1 significant trend (STab), rest descriptive |
| Class imbalance | directional | GBDTs > TabPFN > DL | consistent sign, no sig. trend, confounded severe bin |

**How this advances the thesis.** Raw accuracy is a tie (`results-overview.md`),
so the contribution has to come from the axes where families *do* separate.
Training cost separates them sharply (`cost-analysis.md`); this document adds
that **inference latency** separates them again — and in the *opposite*
direction for TabPFN, which is the cleanest single trade-off in the study
(cheapest to train, most expensive to serve). Size and imbalance separate them
only weakly and with confounds, so the honest framing is: of the five candidate
operational axes, **cost and latency are decisive; size and imbalance are
secondary, confounded, and need more datasets** before they can carry weight.

### Caveats to carry forward

1. Latency is one dataset (`adult`); a multi-dataset latency sweep is the
   obvious Phase-2 hardening.
2. Size axis can't see true large-scale — everything ≥100k is capped to ~100k.
   A genuine scaling study needs the uncapped data (and a TabPFN variant that
   handles it).
3. Imbalance is measured through ROC-AUC rank; PR-AUC / KS / calibration under
   imbalance is the sharper Phase-2 metric, and the severe bin must be
   de-confounded from cardinality (drop or fix `amazon_employee`).
4. Multiclass, size, and DL-strength are mutually confounded (all multiclass
   sets are medium/large). Disentangling them needs small multiclass datasets.
