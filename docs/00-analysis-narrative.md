> **NOTA (18/07/2026):** este documento retrata a fase de **11 modelos** do benchmark. O estado atual (14 modelos, 250/252 células, TabFM líder nas três tarefas) está em `notebooks/00_presentation.ipynb` (§6.2), `notebooks/TABELA_RESULTADOS.md` e `docs/programa-de-pesquisa.md`. Mantido como registro histórico da fase.

# The Benchmark, as a Story — Analytical Narrative & Design

> **What this document is.** Before any plot is drawn, this is the *spine* of the
> analytical phase: the question we are really asking, the four-act story the
> evidence tells, the methodological layer that lifts this from "a table of
> numbers" to a sophisticated study, and an honest map of what we can prove now
> versus what needs fresh compute. Every notebook and every figure exists to
> serve one of the four acts below. Read this first; everything else is detail.
>
> It is written to be **didactic** — each statistical tool is explained in plain
> language the first time it appears (see the Glossary, §8), so the document
> doubles as the methods narrative for Individual Development 1 and the thesis.

---

## 1. The question the field is actually arguing about

For a decade the working assumption in applied ML was blunt: **"on tabular data,
gradient-boosted trees win; deep learning is not worth the trouble."** That
sentence shaped real engineering decisions.

Two things happened in 2024–2025 that make the sentence worth re-testing:

1. **A new generation of deep models** designed *specifically* for tables —
   TabM (batch-ensembled MLPs), RealMLP (an MLP with carefully engineered
   defaults), STab (stochastic competition units) — that are not the clumsy
   adaptations of vision/NLP nets the old verdict was based on.
2. **Foundation models for tables** — TabPFN v2, a transformer *pre-trained on
   millions of synthetic tabular problems* that makes predictions **with no
   training at all**, in a single forward pass. (Published in *Nature*, 2025 —
   the field's hottest result.)

So the thesis question is not the tired "trees vs nets." It is sharper:

> **In 2025, with the modern tabular-DL and foundation-model toolkit, does model
> choice still have a simple answer — and if accuracy no longer separates the
> field, what *should* drive the decision?**

That second clause is the whole thesis. The answer, foreshadowed by our results,
is that the interesting separation has *moved* — out of the accuracy column and
into cost, latency, robustness, and task structure. The contribution is to make
that migration **rigorous, multi-axis, and actionable**.

---

## 2. The four-act narrative

The benchmark is not a list of winners. It is an argument with a beginning,
a complication, a twist, and a resolution.

### Act I — The Tie *(the setup)*

**Claim:** On raw predictive performance, the top of the field is a statistical
dead heat. Across 10 binary and 5 regression datasets, the Friedman test says
"a difference exists somewhere," but the post-hoc tests cannot separate the top
6–7 models — TabPFN, the three GBDTs, TabM, RealMLP all overlap. The *only*
robust separations are at the bottom (TabNet is reliably weakest).

**Why it matters:** This is the load-bearing finding. If accuracy decided the
question, there would be no thesis. Because it *doesn't*, the decision must be
made on other grounds — which is the entire rest of the story. We make this
claim **honestly and powerfully** by confronting the low statistical power
head-on (only 18 datasets) with the right tools (§4): not by hiding it, but by
quantifying exactly how indistinguishable the top cluster is.

> Didactic hook: "The headline isn't *who won*. It's that *nobody won by enough
> to notice* — and that emptiness is the most useful result in the study."

### Act II — The Hidden Axes *(the complication)*

**Claim:** The models that tie on accuracy are *wildly* different in cost. The
spread in training time is ~300×; the spread in inference latency is ~26,000×.
And the rankings on these axes are almost the *inverse* of intuition.

**The emblematic finding — TabPFN's inversion:** the model that is *cheapest to
train* (≈1 second, zero tuning) is the *second most expensive to serve* (72
seconds to score 10k rows, ~22,000× slower than XGBoost). Its whole cost
identity is upside-down. No accuracy table can show this; it only appears when
you put cost on its own axis.

**Why it matters:** Cost is the first axis where the families *do* separate
cleanly — and it separates them in a way that is directly actionable for a
practitioner. This is where "it depends" starts to get a precise meaning.

### Act III — The Reversals *(the twist)*

**Claim:** Even accuracy is not really one number — it is *conditional*. The best
family is a **function of the problem's structure**:

- **By task:** GBDTs/TabPFN lead binary & regression, but **deep learning
  reverses ahead on multiclass** (STab, TabM, RealMLP top all three multiclass
  sets). The verdict literally flips with the task type.
- **By size:** GBDTs improve their standing as data grows; STab improves
  significantly; TabPFN stays strong but is capped by its subsample limit.
- **By imbalance:** GBDTs get *relatively stronger* as the minority class gets
  rarer; attention-DL gets weaker.
- **By feature type:** the one categorical-heavy dataset breaks the attention
  models (one-hot explosion) — a pipeline failure masquerading as a model
  failure, and a direct lead for Individual Development 2.

**Why it matters:** This kills "one model to rule them all" with evidence, and it
is what turns a benchmark into a *decision problem*. Each reversal is a branch in
the final flowchart. It also surfaces the **meta-question**: can we *predict*
which family will win from a dataset's measurable properties (size, imbalance,
dimensionality, feature mix)? That is a light meta-learning result and the
intellectual bridge to Act IV.

### Act IV — The Decision Framework *(the resolution)*

**Claim:** Synthesize the axes into the artifact that does not yet exist in the
literature for this 2025 model set: a **multi-criteria decision matrix**
(model × 10 criteria, scored) and a **practitioner flowchart** ("answer 3
questions about your problem → get a defensible shortlist"). Backed by the most
rigorous benchmark of these models to date (nested CV, Bayesian + frequentist
tests, cost on real hardware).

**Why it matters:** This is the concrete, citable contribution of Individual
Development 1 and the backbone of the thesis. It also *names where the novel
contribution (AI-2) lives* — the categorical-encoding gap for DL, the
imbalance fragility, or the TabPFN train/serve inversion are each a candidate
problem with a clear hook.

---

## 3. The ten criteria, mapped to the story

The professor's guidance was to **segment the analysis across orthogonal axes**.
Here is every axis, which act it serves, and — crucially — **whether the
evidence already exists** or needs fresh compute. (This honesty is itself part
of the rigor.)

| # | Criterion | Act | Evidence status |
|---|-----------|-----|-----------------|
| C1 | Predictive performance | I | ✅ complete (test_results) |
| C2 | Dataset-size sensitivity | III | 🟡 partial — *rank-vs-size yes; true learning curves need re-runs* |
| C3 | Feature dimensionality | III | ✅ computable now (meta) |
| C4 | Feature-type sensitivity | III | ✅ computable now (meta) |
| C5 | Class-imbalance robustness | III | ✅ computable now |
| C6 | Computational cost | II | ✅ complete |
| C7 | Inference latency | II | ✅ measured (adult); 🟡 *multi-dataset sweep is a stretch goal* |
| C8 | Tuning sensitivity | (meta) | 🔴 needs re-runs with defaults (Optuna trials were not stored) |
| C9 | Interpretability | IV | ✅ computable now (GBDT/SHAP); DL-SHAP optional |
| C10 | Practical robustness | III | ✅ computable now (fold_results) |

**The discipline:** we do not pretend C2-curves and C8 exist. We compute the
8 axes the finished benchmark already supports — richly — and scope C2-full and
C8 as **named next experiments** (and natural AI-2 seeds). A reviewer trusts a
study that draws its own boundary more than one that blurs it.

---

## 4. The sophistication layer — what lifts this above a vanilla benchmark

A benchmark that reports mean accuracy and a bar chart is a homework set. These
additions are what make it a *thesis*. All are computable from data we already
have.

### 4.1 Confront low power honestly — Bayesian model comparison
With only 18 datasets, classical null-hypothesis tests (Friedman/Nemenyi) are
**under-powered**: "not significant" gets misread as "no difference." We add the
**Bayesian signed-rank test** (Benavoli et al., 2017), which answers the
question a practitioner actually has — *"what is the probability that model A is
better than B?"* — and includes a **region of practical equivalence (ROPE)**:
the band within which two models are "practically the same." Instead of a binary
reject/don't-reject, we report `P(A≫B) / P(rope) / P(B≫A)`. This turns the
low-power weakness into a *feature*: a calibrated statement of uncertainty.
*Serves Act I; it is the rigorous way to say "it's a tie."*

### 4.2 Normalize before aggregating — distance-to-best
Averaging ROC-AUC across datasets is sloppy (a 0.02 swing means different things
on different datasets). We report, per model, the **normalized regret** —
distance to the best model on each dataset, scaled to [0,1] — and aggregate
*that*. It is the scale-invariant way to say "how much do you lose, on average,
by picking this model instead of the per-dataset oracle." *Serves Acts I & IV.*

### 4.3 Is "family" even the right unit? — variance decomposition
A quietly powerful question (framework C1.6): **is the spread *within* a family
larger than the spread *between* families?** If yes, "GBDT vs DL" is the wrong
abstraction and we should talk about individual models. We decompose rank
variance into between-family and within-family components (a one-way
ANOVA-style split). *Serves Act I; sharpens every family-level claim.*

### 4.4 Can we *predict* the winner? — meta-feature analysis
The bridge from "reversals" to "flowchart." For each dataset we compute
**meta-features** (n_samples, n_features, imbalance ratio, categorical fraction,
task type) and ask which of them predict the winning *family*. Two passes:
(a) descriptive — correlation of each meta-feature with each model's rank;
(b) a tiny interpretable decision tree `meta-features → winning family`. The
tree's splits *are* the flowchart's branches, now data-derived rather than
hand-waved. This is a genuine (if small-N) meta-learning result. *Serves Acts
III & IV.*

### 4.5 Average-case is not enough — risk & robustness
A model that is great on average but occasionally catastrophic is dangerous in
production. From the per-fold results we add: **rank stability** (std of a
model's rank across CV folds), **worst-case rank** (its floor), and a
**catastrophic-failure count** (folds > 2σ below its own mean). This reframes
selection as *risk-adjusted*, not just mean-optimal. *Serves Act III; criterion
C10.*

### 4.6 Are the probabilities trustworthy? — calibration *(optional pass)*
TabPFN is pre-trained on *synthetic* data; its probabilities may be
miscalibrated on real problems — which matters more than AUC in finance/health.
We can run a focused calibration pass (reliability curves + ECE) on a few
datasets, mirroring the latency pass. *Scoped as optional — needs a short
re-inference run; high payoff for the regulated-industry angle.*

---

## 5. What we run now vs. what we name as next compute

**Run now (no retraining — all from the finished benchmark):**
C1, C3, C4, C5, C6, C7(adult), C9, C10, plus every item in §4.1–4.5, plus the
decision matrix + flowchart (Act IV). This is a complete, sophisticated IA-1.

**Named as next experiments (honest scope, AI-2 seeds):**
- **C2 true learning curves** — retrain each model on {10,20,40,60,80,100}%
  fractions to find the *crossover* sample size where DL catches GBDTs
  (Research Question 2). Real GPU cost; highest-value next run.
- **C8 tuning sensitivity** — re-run each model once with paper defaults to
  measure the *value of tuning* per model (Optuna trials weren't persisted).
- **C6/C7 calibration & multi-dataset latency** — short re-inference passes.

---

## 6. Deliverable map

| Notebook / doc | Story role | Action |
|----------------|-----------|--------|
| 01 data exploration | sets the stage (the 18 datasets) | fix stale "15"→18 |
| 02 results overview | Act I — heatmaps, ranks, win matrix | fix; add normalized-regret + variance decomposition |
| 03 statistical tests | Act I — Friedman/Nemenyi/Wilcoxon/CD | fix stale text; **add Bayesian signed-rank + ROPE** |
| 04 scaling | Act III — size & dimensionality | fix metric (use primary, not accuracy); demote learning-curve stub to scoped note |
| 05 cost | Act II — training/tuning/Pareto | **wire in measured latency**; fix Pareto metric |
| 06 interpretability | Act IV input — importance/SHAP/agreement | fix stale dataset list |
| **07 segmented robustness** *(new)* | Act III — imbalance C5 + feature-type C4 + risk C10 | build from `segmented-analysis.md` + fold_results |
| **08 meta-features** *(new)* | Act III→IV bridge — predict-the-winner | build (§4.4) |
| **09 decision framework** *(new)* | Act IV — matrix + flowchart | build (the contribution) |
| `00-analysis-narrative.md` (this) | the spine | — |
| `results-overview / cost / segmented` (.md) | written companions | already done; align numbers |

---

## 7. The arc in one paragraph (for the abstract / the call)

> We benchmark 11 models spanning GBDTs, modern tabular deep learning, and a
> foundation model across 18 OpenML datasets under nested cross-validation. On
> raw performance the top of the field is statistically inseparable — a result
> we establish with both frequentist and Bayesian tests rather than gloss over.
> The separation that matters has moved off the accuracy axis: training cost
> spans ~300×, inference latency ~26,000×, and the cheapest-to-train model
> (TabPFN) is among the most expensive to serve — an inversion invisible to any
> accuracy table. Performance itself proves *conditional*: the leading family
> flips with task type, dataset size, class imbalance, and feature composition,
> and these dependencies are partly *predictable* from dataset meta-features. We
> synthesize the axes into a multi-criteria decision matrix and a practitioner
> flowchart — the first for the 2024–2025 model generation — and identify the
> concrete limitation (DL's categorical-encoding bottleneck) that the next phase
> will address.

---

## 8. Glossary (plain-language, first-use reference)

- **Nested cross-validation** — an outer loop to *measure* a model honestly and
  an inner loop to *tune* it, so tuning never peeks at the test data. Prevents
  the optimism of tuning and testing on the same split.
- **Friedman test** — "across all these datasets, is *any* model reliably
  different from the others?" One yes/no for the whole table.
- **Nemenyi / critical difference (CD)** — after Friedman says "yes," *which*
  pairs differ. The CD is the rank gap two models must exceed to count as
  different; on a CD diagram, models joined by a bar are statistically tied.
- **Wilcoxon signed-rank (Holm-corrected)** — a pairwise "is A better than B
  across datasets?", with a correction for testing many pairs at once.
- **Cohen's d** — *how big* a difference is (effect size), independent of
  whether it's statistically significant. Small N can hide a real effect;
  large N can flag a trivial one. d tells them apart.
- **Bayesian signed-rank + ROPE** — instead of reject/don't-reject, the
  *probability* A is better than B, with a "region of practical equivalence"
  for "effectively the same." The honest tool when you have few datasets.
- **Pareto frontier** — the set of models that are not *dominated* (nobody is
  both cheaper *and* better). The rational shortlist on a cost-vs-performance
  plot.
- **Normalized regret / distance-to-best** — how much you lose, per dataset, by
  not having picked that dataset's best model; scale-invariant, so it can be
  averaged across datasets fairly.
- **Meta-feature** — a measurable property *of a dataset* (size, imbalance,
  dimensionality, % categorical) used to predict which model will do well.
- **Calibration / ECE** — do predicted probabilities mean what they say? (Of the
  cases a model calls "80% positive," are ~80% actually positive?) Distinct from
  accuracy and critical in regulated domains.
